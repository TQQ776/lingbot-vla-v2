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
    "marker_displacement_history",
    "marker_valid_mask",
    "marker_history_valid_mask",
    "tactile_sensor_mask",
    "tactile_rgb_mask",
)


def tactile_dataset_keys(settings: "TactileVTLAConfig") -> set[str]:
    """Return every LeRobot feature that may supply enabled tactile inputs."""

    keys: set[str] = set()
    if settings.use_rgb:
        keys.update(settings.rgb_keys)
    if settings.use_markers:
        keys.update(settings.marker_displacement_keys)
        keys.update(settings.marker_valid_mask_keys)
    return keys


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def left_pad_marker_history(
    value: Any,
    *,
    history_length: int,
    trailing_shape: tuple[int, ...],
    name: str,
    dtype: torch.dtype,
) -> Tensor:
    """Return the latest history in oldest-to-newest order with earliest replication."""

    tensor = _as_tensor(value, dtype=dtype)
    if tuple(tensor.shape) == trailing_shape:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != len(trailing_shape) + 1 or tuple(tensor.shape[1:]) != trailing_shape:
        raise ValueError(
            f"{name} must be {trailing_shape} or [T,{','.join(map(str, trailing_shape))}], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] == 0:
        raise ValueError(f"{name} history must not be empty")
    tensor = tensor[-history_length:]
    if tensor.shape[0] < history_length:
        pad = tensor[:1].expand(history_length - tensor.shape[0], *trailing_shape)
        tensor = torch.cat([pad, tensor], dim=0)
    return tensor.contiguous()


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

    Marker datasets store one normalized displacement frame per physical row.
    LeRobot supplies up to ``marker_history_length`` same-episode rows, and this
    transform left-pads short episode prefixes by replicating their first frame.
    """

    if not settings.enabled:
        return {}

    num_sensors = settings.num_sensors
    num_markers = settings.num_markers
    history_length = settings.marker_history_length
    sensor_present = torch.zeros(num_sensors, dtype=torch.bool)
    result: dict[str, Tensor] = {}

    if settings.use_markers:
        aggregate_history = item.get("marker_displacement_history")
        aggregate_valid = item.get("marker_valid_mask")
        aggregate_history_valid = item.get("marker_history_valid_mask")
        if aggregate_history is not None:
            aggregate_history = _as_tensor(aggregate_history, dtype=torch.float32)
            expected_history_shape = (
                num_sensors,
                history_length,
                num_markers,
                2,
            )
            if tuple(aggregate_history.shape) != expected_history_shape:
                raise ValueError(
                    "marker_displacement_history must have shape "
                    f"{expected_history_shape}"
                )
            if aggregate_valid is not None:
                aggregate_valid = _as_tensor(aggregate_valid).to(dtype=torch.bool)
                if tuple(aggregate_valid.shape) != expected_history_shape[:-1]:
                    raise ValueError("marker_valid_mask must have shape [S,H,N]")
            if aggregate_history_valid is not None:
                aggregate_history_valid = _as_tensor(aggregate_history_valid).to(
                    dtype=torch.bool
                )
                if tuple(aggregate_history_valid.shape) != (
                    num_sensors,
                    history_length,
                ):
                    raise ValueError("marker_history_valid_mask must have shape [S,H]")
        histories: list[Tensor] = []
        valid_masks: list[Tensor] = []
        history_valid_masks: list[Tensor] = []
        for sensor_index in range(num_sensors):
            displacement_key = settings.marker_displacement_keys[sensor_index]
            valid_key = settings.marker_valid_mask_keys[sensor_index]

            if aggregate_history is not None:
                history = aggregate_history[sensor_index]
                sensor_present[sensor_index] = True
            elif displacement_key in item:
                history = left_pad_marker_history(
                    item[displacement_key],
                    history_length=history_length,
                    trailing_shape=(num_markers, 2),
                    name=displacement_key,
                    dtype=torch.float32,
                )
                sensor_present[sensor_index] = True
            else:
                history = torch.zeros(
                    history_length, num_markers, 2, dtype=torch.float32
                )

            if tuple(history.shape) != (history_length, num_markers, 2):
                raise ValueError(
                    f"Sensor {sensor_index} marker history must be "
                    f"[{history_length},{num_markers},2], got {tuple(history.shape)}"
                )
            if aggregate_valid is not None:
                valid = aggregate_valid[sensor_index]
            elif valid_key in item:
                valid = left_pad_marker_history(
                    item[valid_key],
                    history_length=history_length,
                    trailing_shape=(num_markers,),
                    name=valid_key,
                    dtype=torch.bool,
                )
            else:
                valid = torch.isfinite(history).all(dim=-1)
                valid &= bool(sensor_present[sensor_index])
            if aggregate_history_valid is not None:
                history_valid = aggregate_history_valid[sensor_index]
            else:
                history_valid = torch.full(
                    (history_length,),
                    bool(sensor_present[sensor_index]),
                    dtype=torch.bool,
                )
            if not torch.isfinite(history).all():
                raise ValueError(f"Sensor {sensor_index} marker values must be finite")
            histories.append(history)
            valid_masks.append(valid)
            history_valid_masks.append(history_valid)

        result.update(
            marker_displacement_history=torch.stack(histories, dim=0),
            marker_valid_mask=torch.stack(valid_masks, dim=0),
            marker_history_valid_mask=torch.stack(history_valid_masks, dim=0),
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
            result["marker_valid_mask"] &= explicit_sensor_mask[:, None, None]
            result["marker_history_valid_mask"] &= explicit_sensor_mask[:, None]
        if "tactile_rgb_mask" in result:
            result["tactile_rgb_mask"] &= explicit_sensor_mask
    result["tactile_sensor_mask"] = sensor_present
    return result


__all__ = [
    "MODEL_TACTILE_KEYS",
    "left_pad_marker_history",
    "prepare_tactile_sample",
    "tactile_dataset_keys",
]
