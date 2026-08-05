"""TacThru sample preparation for the minimal LingBot-VLA VTLA path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

if TYPE_CHECKING:
    from ...models.vla.lingbot_vla.tactile_vtla import TactileVTLAConfig


MODEL_TACTILE_KEYS = (
    "tactile_rgb",
    "tactile_rgb_grid_thw",
    "marker_positions",
    "marker_reference",
    "previous_marker_positions",
    "marker_valid_mask",
    "tactile_sensor_mask",
    "tactile_rgb_mask",
)


def tactile_dataset_keys(settings: "TactileVTLAConfig") -> set[str]:
    """Return every LeRobot feature that may supply enabled tactile inputs."""

    keys: set[str] = set()
    if settings.use_rgb:
        keys.update(settings.rgb_keys)
    if settings.use_markers:
        keys.update(settings.marker_positions_keys)
        keys.update(settings.marker_reference_keys)
        keys.update(settings.marker_valid_mask_keys)
        keys.update(settings.marker_flow_keys)
    return keys


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _current_and_previous(value: Any, *, name: str) -> tuple[Tensor, Tensor]:
    tensor = _as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 2 and tensor.shape[-1] == 2:
        return tensor, tensor
    if tensor.ndim == 3 and tensor.shape[-1] == 2 and tensor.shape[0] > 0:
        return tensor[-1], tensor[-2] if tensor.shape[0] > 1 else tensor[-1]
    raise ValueError(f"{name} must be [N,2] or [T,N,2], got {tuple(tensor.shape)}")


def _current_reference(value: Any, *, name: str) -> Tensor:
    tensor = _as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 2 and tensor.shape[-1] == 2:
        return tensor
    if tensor.ndim == 3 and tensor.shape[-1] == 2 and tensor.shape[0] > 0:
        return tensor[-1]
    raise ValueError(f"{name} must be [N,2] or [T,N,2], got {tuple(tensor.shape)}")


def _current_valid_mask(value: Any, *, num_markers: int, name: str) -> Tensor:
    mask = _as_tensor(value).to(dtype=torch.bool)
    if mask.ndim == 1:
        result = mask
    elif mask.ndim == 2 and mask.shape[0] > 0:
        result = mask[-1]
    else:
        raise ValueError(f"{name} must be [N] or [T,N], got {tuple(mask.shape)}")
    if tuple(result.shape) != (num_markers,):
        raise ValueError(f"{name} must contain {num_markers} markers")
    return result


def _last_rgb_frame(value: Any, *, name: str) -> Tensor:
    image = _as_tensor(value)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3:
        raise ValueError(f"{name} must be [C,H,W] or [T,C,H,W], got {tuple(image.shape)}")
    if image.shape[0] != 3 and image.shape[-1] == 3:
        image = image.permute(2, 0, 1).contiguous()
    if image.shape[0] != 3:
        raise ValueError(f"{name} must have three RGB channels")
    return image


def _process_qwen_image(image_processor: Any, image: Tensor) -> tuple[Tensor, Tensor]:
    if image_processor is None:
        raise ValueError("Tactile RGB requires the Qwen image processor")
    processed = image_processor(image)
    if "pixel_values" not in processed or "image_grid_thw" not in processed:
        raise ValueError("Qwen image processor must return pixel_values and image_grid_thw")
    pixels = _as_tensor(processed["pixel_values"])
    grid = _as_tensor(processed["image_grid_thw"]).to(dtype=torch.long).reshape(-1, 3)
    if grid.shape[0] != 1:
        raise ValueError("Each tactile RGB sensor must produce exactly one image grid")
    if pixels.ndim >= 3 and pixels.shape[0] == 1:
        pixels = pixels.squeeze(0)
    return pixels, grid[0]


def prepare_tactile_sample(
    item: Mapping[str, Any],
    image_processor: Any,
    settings: "TactileVTLAConfig",
    *,
    fallback_image_keys: tuple[str, ...] = (),
) -> dict[str, Tensor]:
    """Create fixed-shape model tactile tensors for one sample.

    Position/reference fields are preferred.  Existing TacThru datasets that
    only store normalized marker flow are represented equivalently as
    ``position=flow`` and ``reference=0``.  Their previous queried flow becomes
    ``previous_marker_positions``, preserving both displacement and velocity.
    """

    if not settings.enabled:
        return {}

    num_sensors = settings.num_sensors
    num_markers = settings.num_markers
    sensor_present = torch.zeros(num_sensors, dtype=torch.bool)
    result: dict[str, Tensor] = {}

    if settings.use_markers:
        aggregate_positions = item.get("marker_positions")
        aggregate_reference = item.get("marker_reference")
        aggregate_previous = item.get("previous_marker_positions")
        aggregate_valid = item.get("marker_valid_mask")
        if aggregate_positions is not None:
            aggregate_positions = _as_tensor(aggregate_positions, dtype=torch.float32)
            if tuple(aggregate_positions.shape) != (num_sensors, num_markers, 2):
                raise ValueError("marker_positions must have shape [S,N,2]")
            if aggregate_reference is None:
                raise KeyError("marker_positions requires marker_reference")
            aggregate_reference = _as_tensor(aggregate_reference, dtype=torch.float32)
            if tuple(aggregate_reference.shape) != tuple(aggregate_positions.shape):
                raise ValueError("marker_reference must have shape [S,N,2]")
            if aggregate_previous is not None:
                aggregate_previous = _as_tensor(
                    aggregate_previous, dtype=torch.float32
                )
                if tuple(aggregate_previous.shape) != tuple(aggregate_positions.shape):
                    raise ValueError("previous_marker_positions must have shape [S,N,2]")
            if aggregate_valid is not None:
                aggregate_valid = _as_tensor(aggregate_valid).to(dtype=torch.bool)
                if tuple(aggregate_valid.shape) != (num_sensors, num_markers):
                    raise ValueError("marker_valid_mask must have shape [S,N]")
        positions: list[Tensor] = []
        references: list[Tensor] = []
        previous: list[Tensor] = []
        valid_masks: list[Tensor] = []
        for sensor_index in range(num_sensors):
            pos_key = settings.marker_positions_keys[sensor_index]
            ref_key = settings.marker_reference_keys[sensor_index]
            valid_key = settings.marker_valid_mask_keys[sensor_index]
            flow_key = settings.marker_flow_keys[sensor_index]

            if aggregate_positions is not None:
                current = aggregate_positions[sensor_index]
                reference = aggregate_reference[sensor_index]
                previous_position = (
                    current
                    if aggregate_previous is None
                    else aggregate_previous[sensor_index]
                )
                sensor_present[sensor_index] = True
            elif pos_key in item:
                if ref_key not in item:
                    raise KeyError(f"{pos_key!r} requires {ref_key!r}")
                current, previous_position = _current_and_previous(
                    item[pos_key], name=pos_key
                )
                reference = _current_reference(item[ref_key], name=ref_key)
                sensor_present[sensor_index] = True
            elif flow_key in item:
                current, previous_position = _current_and_previous(
                    item[flow_key], name=flow_key
                )
                reference = torch.zeros_like(current)
                sensor_present[sensor_index] = True
            else:
                current = torch.zeros(num_markers, 2, dtype=torch.float32)
                reference = torch.zeros_like(current)
                previous_position = torch.zeros_like(current)

            if tuple(current.shape) != (num_markers, 2):
                raise ValueError(
                    f"Sensor {sensor_index} marker shape must be [{num_markers},2], "
                    f"got {tuple(current.shape)}"
                )
            if aggregate_valid is not None:
                valid = aggregate_valid[sensor_index]
            elif valid_key in item:
                valid = _current_valid_mask(
                    item[valid_key],
                    num_markers=num_markers,
                    name=valid_key,
                )
            else:
                valid = torch.isfinite(current).all(dim=-1)
                valid &= bool(sensor_present[sensor_index])
            if not (
                torch.isfinite(current).all()
                and torch.isfinite(reference).all()
                and torch.isfinite(previous_position).all()
            ):
                raise ValueError(f"Sensor {sensor_index} marker values must be finite")
            positions.append(current)
            references.append(reference)
            previous.append(previous_position)
            valid_masks.append(valid)

        result.update(
            marker_positions=torch.stack(positions, dim=0),
            marker_reference=torch.stack(references, dim=0),
            previous_marker_positions=torch.stack(previous, dim=0),
            marker_valid_mask=torch.stack(valid_masks, dim=0),
        )

    if settings.use_rgb:
        aggregate_rgb = item.get("tactile_rgb")
        if aggregate_rgb is not None:
            aggregate_rgb = _as_tensor(aggregate_rgb)
            if aggregate_rgb.ndim != 4 or aggregate_rgb.shape[0] != num_sensors:
                raise ValueError("Raw tactile_rgb must have shape [S,C,H,W] or [S,H,W,C]")
        rgb_frames: list[Tensor | None] = []
        template: Tensor | None = None
        for key in fallback_image_keys:
            if key in item:
                template = _last_rgb_frame(item[key], name=key)
                break
        for sensor_index, key in enumerate(settings.rgb_keys):
            if aggregate_rgb is not None:
                frame = _last_rgb_frame(
                    aggregate_rgb[sensor_index], name="tactile_rgb"
                )
                template = frame if template is None else template
                sensor_present[sensor_index] = True
                rgb_frames.append(frame)
            elif key in item:
                frame = _last_rgb_frame(item[key], name=key)
                template = frame if template is None else template
                sensor_present[sensor_index] = True
                rgb_frames.append(frame)
            else:
                rgb_frames.append(None)
        if template is None:
            raise ValueError(
                "Tactile RGB is enabled but neither tactile nor fallback scene RGB is available"
            )

        processed_images: list[Tensor] = []
        grids: list[Tensor] = []
        rgb_valid: list[bool] = []
        for frame in rgb_frames:
            is_valid = frame is not None
            if frame is None:
                frame = torch.zeros_like(template)
            elif frame.shape[-2:] != template.shape[-2:]:
                original_dtype = frame.dtype
                resized = F.interpolate(
                    frame.unsqueeze(0).to(torch.float32),
                    size=template.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if original_dtype == torch.uint8:
                    frame = resized.round().clamp(0, 255).to(torch.uint8)
                else:
                    frame = resized.to(original_dtype)
            pixels, grid = _process_qwen_image(image_processor, frame)
            processed_images.append(pixels)
            grids.append(grid)
            rgb_valid.append(is_valid)
        shapes = {tuple(value.shape) for value in processed_images}
        if len(shapes) != 1:
            raise ValueError(f"Tactile RGB processor returned inconsistent shapes: {shapes}")
        result.update(
            tactile_rgb=torch.stack(processed_images, dim=0),
            tactile_rgb_grid_thw=torch.stack(grids, dim=0),
            tactile_rgb_mask=torch.tensor(rgb_valid, dtype=torch.bool),
        )

    explicit_sensor_mask = item.get("tactile_sensor_mask")
    if explicit_sensor_mask is not None:
        explicit_sensor_mask = _as_tensor(explicit_sensor_mask).to(dtype=torch.bool)
        if tuple(explicit_sensor_mask.shape) != (num_sensors,):
            raise ValueError("tactile_sensor_mask must have shape [S]")
        sensor_present &= explicit_sensor_mask
        if "marker_valid_mask" in result:
            result["marker_valid_mask"] &= explicit_sensor_mask[:, None]
        if "tactile_rgb_mask" in result:
            result["tactile_rgb_mask"] &= explicit_sensor_mask
    result["tactile_sensor_mask"] = sensor_present
    return result


__all__ = [
    "MODEL_TACTILE_KEYS",
    "prepare_tactile_sample",
    "tactile_dataset_keys",
]
