from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import click
import cv2
import numpy as np
import zarr
from numcodecs import Blosc

from diffusion_policy.common.replay_buffer import ReplayBuffer
from supp.pipeline_utils import resolve_session_from_any, tactile_dataset_key_map
from utils.cv_utils import FisheyeRectConverter, draw_predefined_mask, get_image_transform, parse_fisheye_intrinsics


NUM_CHECKPOINTS = np.array([8, 8])
CENTER_RATIOS = {"tacthru": np.array([0.49860856, 0.49222472]), "TacThru": np.array([0.510953, 0.4904979])}
MAX_RATIOS = {"tacthru": np.array([0.922359, 0.9284391]), "TacThru": np.array([0.90615195, 0.8923828])}
MIN_RATIOS = {"tacthru": np.array([0.07485813, 0.05601038]), "TacThru": np.array([0.11575411, 0.08861297])}


def compute_sensor_padding(orig_img_size_hw: tuple[int, int], sensor_type: str) -> tuple[int, int, int, int]:
    height, width = int(orig_img_size_hw[0]), int(orig_img_size_hw[1])
    orig_img_size = np.array([height, width], dtype=np.int32)

    center_ratio = CENTER_RATIOS[sensor_type]
    max_ratio = MAX_RATIOS[sensor_type]
    min_ratio = MIN_RATIOS[sensor_type]

    center_point = center_ratio * orig_img_size
    ckpt_spacing = (max_ratio - min_ratio) * orig_img_size / (NUM_CHECKPOINTS - 1)
    half_span = NUM_CHECKPOINTS / 2 + 0.5
    padding_rb = (center_point + ckpt_spacing * half_span) - orig_img_size
    padding_lt = -center_point + ckpt_spacing * half_span

    top = int(max(0.0, np.ceil(padding_lt[0])))
    bottom = int(max(0.0, np.ceil(padding_rb[0])))
    left = int(max(0.0, np.ceil(padding_lt[1])))
    right = int(max(0.0, np.ceil(padding_rb[1])))
    return top, bottom, left, right


def _load_marker_flow(kpts_path: Path) -> np.ndarray:
    with kpts_path.open("rb") as f:
        data = pickle.load(f)
    if "marker_flow" in data:
        return np.asarray(data["marker_flow"], dtype=np.float32)
    marker = np.asarray(data["marker"], dtype=np.float32)
    marker_ref = np.asarray(data["marker_ref"], dtype=np.float32)
    return marker - marker_ref


def _infer_marker_count(video_paths: list[str]) -> int:
    for video_path_str in video_paths:
        demo_dir = Path(video_path_str).parent
        for sensor_idx in (0, 1):
            kpts_path = demo_dir / f"TacThru-{sensor_idx}_synced_kpts.pkl"
            if not kpts_path.is_file():
                continue
            marker_flow = _load_marker_flow(kpts_path)
            if marker_flow.ndim != 3 or marker_flow.shape[-1] != 2:
                raise RuntimeError(f"Invalid marker flow shape in {kpts_path}: expected (T, N, 2), got {marker_flow.shape}")
            return int(marker_flow.shape[1])
    raise click.ClickException("Failed to infer tactile marker count: no synced tactile keypoint files were found.")


def _video_shape(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def _load_tactile_sensor_names(demo_dir: Path) -> list[str] | None:
    meta_path = demo_dir / "tactile_meta.json"
    if not meta_path.is_file():
        return None
    with meta_path.open("r") as f:
        payload = json.load(f)
    sensor_names = payload.get("sensor_names")
    if not isinstance(sensor_names, list):
        return None
    return [str(name) for name in sensor_names]


def _dataset_base_from_sensor_name(sensor_name: str, swap_lr: bool) -> str | None:
    normalized = sensor_name.strip().lower()
    if normalized.endswith("l") or normalized.endswith("_left") or normalized.endswith("-left"):
        base = "tacthru_l"
    elif normalized.endswith("r") or normalized.endswith("_right") or normalized.endswith("-right"):
        base = "tacthru_r"
    else:
        return None
    if not swap_lr:
        return base
    return "tacthru_r" if base == "tacthru_l" else "tacthru_l"


def _tactile_dataset_base(demo_dir: Path, sensor_idx: int, swap_lr: bool) -> str:
    sensor_names = _load_tactile_sensor_names(demo_dir)
    if sensor_names is not None and sensor_idx < len(sensor_names):
        base = _dataset_base_from_sensor_name(sensor_names[sensor_idx], swap_lr=swap_lr)
        if base is not None:
            return base
    return tactile_dataset_key_map(swap_lr=swap_lr)[sensor_idx]


def _available_tactile_streams(demo_dir: Path, swap_lr: bool) -> dict[int, str]:
    streams: dict[int, str] = {}
    for sensor_idx in (0, 1):
        if (demo_dir / f"TacThru-{sensor_idx}_synced.mp4").is_file():
            streams[sensor_idx] = _tactile_dataset_base(demo_dir=demo_dir, sensor_idx=sensor_idx, swap_lr=swap_lr)
    return streams


def _ordered_existing_keys(keys: set[str], canonical_order: tuple[str, ...]) -> list[str]:
    ordered = [key for key in canonical_order if key in keys]
    ordered.extend(sorted(keys.difference(ordered)))
    return ordered


@click.command(help="Generate the final zipped zarr dataset from dataset_plan_vive.pkl and the synced videos.")
@click.argument("input", nargs=-1, required=True, type=click.Path(path_type=Path, exists=True))
@click.option("-o", "--output", required=True, type=click.Path(path_type=Path))
@click.option("-or", "--out-res", type=str, default="224,224", show_default=True)
@click.option("-of", "--out-fov", type=float, default=120.0, show_default=True)
@click.option(
    "--camera-intrinsics",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional fisheye intrinsics JSON. Synria C10 defaults to direct resize.",
)
@click.option("-cl", "--compression-level", type=int, default=5, show_default=True)
@click.option("-nm", "--no-mirror", is_flag=True, default=False, help="Mask out the predefined mirror regions in camera observations.")
@click.option("-ms", "--mirror-swap", is_flag=True, default=False, help="Mirror-swap the predefined camera mirror regions.")
@click.option("--swap-lr/--no-swap-lr", default=False, help="Map TacThru-0/1 to right/left instead of left/right.")
@click.option("-n", "--num-workers", type=int, default=None)
@click.option("--patch", type=click.Choice(["patch14", "patch16"]), default=None)
@click.option("--write-markers/--no-write-markers", default=True, help="Write tactile marker flow tensors.")
@click.option("--force", is_flag=True, default=False)
def main(
    input: tuple[Path, ...],
    output: Path,
    out_res: str,
    out_fov: float,
    camera_intrinsics: Path | None,
    compression_level: int,
    no_mirror: bool,
    mirror_swap: bool,
    swap_lr: bool,
    num_workers: int | None,
    patch: str | None,
    write_markers: bool,
    force: bool,
) -> None:
    _ = num_workers  # The current implementation writes sequentially to avoid replay-buffer races.
    do_padding = patch is not None
    if patch == "patch14":
        patch_size = 14
    elif patch == "patch16":
        patch_size = 16
    else:
        patch_size = None

    if patch_size is not None:
        final_resize = patch_size * (int(NUM_CHECKPOINTS[0]) + 1)
        out_res_xy = (final_resize, final_resize)
    else:
        out_res_xy = tuple(int(x) for x in out_res.split(","))

    output = output.expanduser().resolve()
    if output.exists() and force:
        output.unlink()

    fisheye_converter = None
    if camera_intrinsics is not None:
        with camera_intrinsics.expanduser().resolve().open("r") as f:
            opencv_intr_dict = parse_fisheye_intrinsics(json.load(f))
        fisheye_converter = FisheyeRectConverter(**opencv_intr_dict, out_size=out_res_xy, out_fov=out_fov, principal_point_offset=(0, 0))

    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    videos_dict: dict[str, list[dict]] = defaultdict(list)

    for input_path in input:
        session_dir = resolve_session_from_any(input_path)
        demos_dir = session_dir / "demos"
        plan_path = session_dir / "dataset_plan_vive.pkl"
        if not plan_path.is_file():
            click.echo(f"[dataset] skip {session_dir.name}: missing dataset_plan_vive.pkl")
            continue

        with plan_path.open("rb") as f:
            plan = pickle.load(f)

        buffer_start = replay_buffer.n_steps
        for episode in plan:
            if len(episode["grippers"]) != 1 or len(episode["cameras"]) != 1:
                raise RuntimeError("The pipeline currently supports one robot stream and one camera stream per episode.")

            gripper = episode["grippers"][0]
            tcp_pose = np.asarray(gripper["tcp_pose"], dtype=np.float32)
            episode_data = {
                "robot0_eef_pos": tcp_pose[..., :3].astype(np.float32),
                "robot0_eef_rot_axis_angle": tcp_pose[..., 3:].astype(np.float32),
                "robot0_gripper_width": np.asarray(gripper["gripper_width"], dtype=np.float32).reshape(-1, 1),
                "robot0_demo_start_pose": np.broadcast_to(np.asarray(gripper["demo_start_pose"], dtype=np.float32), tcp_pose.shape).copy(),
                "robot0_demo_end_pose": np.broadcast_to(np.asarray(gripper["demo_end_pose"], dtype=np.float32), tcp_pose.shape).copy(),
            }
            replay_buffer.add_episode(data=episode_data, compressors=None)

            camera = episode["cameras"][0]
            video_path = (demos_dir / camera["video_path"]).resolve()
            frame_start, frame_end = camera["video_start_end"]
            videos_dict[str(video_path)].append(
                {
                    "frame_start": int(frame_start),
                    "frame_end": int(frame_end),
                    "buffer_start": int(buffer_start),
                }
            )
            buffer_start += int(frame_end - frame_start)

    total_steps = replay_buffer.n_steps
    if total_steps <= 0:
        raise click.ClickException("No dataset steps were generated from the dataset plan.")

    tactile_streams_by_demo: dict[str, dict[int, str]] = {}
    tactile_rgb_keys: set[str] = set()
    tactile_marker_keys: set[str] = set()
    for video_path_str in sorted(videos_dict.keys()):
        demo_dir = Path(video_path_str).parent
        streams = _available_tactile_streams(demo_dir=demo_dir, swap_lr=swap_lr)
        if not streams:
            raise FileNotFoundError(f"Missing synced tactile videos for {demo_dir}: expected at least one TacThru-*_synced.mp4")
        tactile_streams_by_demo[str(demo_dir)] = streams
        for dataset_base in streams.values():
            tactile_rgb_keys.add(f"{dataset_base}_rgb")
            tactile_marker_keys.add(f"{dataset_base}_marker")

    marker_count = _infer_marker_count(sorted(videos_dict.keys())) if write_markers else None

    image_compressor = Blosc(cname="zstd", clevel=max(0, min(int(compression_level), 9)), shuffle=Blosc.BITSHUFFLE)
    rgb_keys = ("camera0_rgb", *_ordered_existing_keys(tactile_rgb_keys, ("tacthru_l_rgb", "tacthru_r_rgb")))
    for key in rgb_keys:
        replay_buffer.data.require_dataset(
            name=key,
            shape=(total_steps, out_res_xy[1], out_res_xy[0], 3),
            chunks=(1, out_res_xy[1], out_res_xy[0], 3),
            compressor=image_compressor,
            dtype=np.uint8,
        )
    if write_markers:
        for key in _ordered_existing_keys(tactile_marker_keys, ("tacthru_l_marker", "tacthru_r_marker")):
            replay_buffer.data.require_dataset(
                name=key,
                shape=(total_steps, int(marker_count), 2),
                chunks=(1, int(marker_count), 2),
                compressor=None,
                dtype=np.float32,
            )

    if not videos_dict:
        raise click.ClickException("No camera video tasks were collected from the dataset plan.")

    first_video_path = Path(next(iter(videos_dict.keys())))
    camera_in_res = _video_shape(first_video_path)
    resize_tf = get_image_transform(in_res=camera_in_res, out_res=out_res_xy)

    for video_path_str, tasks in sorted(videos_dict.items()):
        video_path = Path(video_path_str)
        demo_dir = video_path.parent
        tactile_streams = tactile_streams_by_demo[str(demo_dir)]
        sensor_videos = {sensor_idx: demo_dir / f"TacThru-{sensor_idx}_synced.mp4" for sensor_idx in tactile_streams}
        sensor_kpts = {}
        if write_markers:
            for sensor_idx in sensor_videos:
                kpts_path = demo_dir / f"TacThru-{sensor_idx}_synced_kpts.pkl"
                if not kpts_path.is_file():
                    raise FileNotFoundError(f"Missing synced tactile keypoints for {demo_dir}: {kpts_path.name}")
                sensor_kpts[sensor_idx] = _load_marker_flow(kpts_path)
        if write_markers:
            for sensor_idx, marker_flow in sensor_kpts.items():
                if marker_flow is None:
                    continue
                if marker_flow.ndim != 3 or marker_flow.shape[-1] != 2:
                    raise RuntimeError(
                        f"Invalid marker flow shape in {demo_dir / f'TacThru-{sensor_idx}_synced_kpts.pkl'}: "
                        f"expected (T, {marker_count}, 2), got {marker_flow.shape}"
                    )
                if marker_flow.shape[1] != marker_count:
                    raise RuntimeError(
                        f"Inconsistent tactile marker count in {demo_dir / f'TacThru-{sensor_idx}_synced_kpts.pkl'}: "
                        f"expected {marker_count}, got {marker_flow.shape[1]}"
                    )

        camera_cap = cv2.VideoCapture(str(video_path))
        sensor_caps = {idx: cv2.VideoCapture(str(path)) for idx, path in sensor_videos.items()}
        if not camera_cap.isOpened():
            raise RuntimeError(f"Failed to open {video_path}")
        for idx, cap in sensor_caps.items():
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {sensor_videos[idx]}")

        try:
            for task in sorted(tasks, key=lambda item: item["frame_start"]):
                frame_start = task["frame_start"]
                frame_end = task["frame_end"]
                buffer_start = task["buffer_start"]
                camera_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                for cap in sensor_caps.values():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

                for offset, frame_idx in enumerate(range(frame_start, frame_end)):
                    ok, camera_frame = camera_cap.read()
                    if not ok:
                        raise RuntimeError(f"Failed to read camera frame {frame_idx} from {video_path}")
                    camera_frame = camera_frame[..., ::-1]
                    if fisheye_converter is not None:
                        camera_frame = fisheye_converter.forward(camera_frame)
                    else:
                        camera_frame = resize_tf(camera_frame)
                    if no_mirror:
                        camera_frame = draw_predefined_mask(camera_frame, color=(0, 0, 0), mirror=True, gripper=False, finger=False)
                    if mirror_swap:
                        mirror_mask = np.ones_like(camera_frame, dtype=np.uint8)
                        mirror_mask = draw_predefined_mask(mirror_mask, color=(0, 0, 0), mirror=True, gripper=False, finger=False)
                        is_mirror = mirror_mask[..., 0] == 0
                        camera_frame[is_mirror] = camera_frame[:, ::-1, :][is_mirror]
                    replay_buffer.data["camera0_rgb"][buffer_start + offset] = camera_frame

                    for sensor_idx, sensor_cap in sensor_caps.items():
                        ok, sensor_frame = sensor_cap.read()
                        if not ok:
                            raise RuntimeError(f"Failed to read tactile frame {frame_idx} from {sensor_videos[sensor_idx]}")
                        sensor_frame = sensor_frame[..., ::-1]
                        if do_padding:
                            sensor_type = "tacthru" if tactile_streams[sensor_idx] == "tacthru_l" else "TacThru"
                            top, bottom, left, right = compute_sensor_padding(sensor_frame.shape[:2], sensor_type)
                            if top or bottom or left or right:
                                sensor_frame = cv2.copyMakeBorder(
                                    sensor_frame,
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    borderType=cv2.BORDER_REPLICATE,
                                )
                        if (sensor_frame.shape[1], sensor_frame.shape[0]) != out_res_xy:
                            sensor_frame = cv2.resize(sensor_frame, out_res_xy, interpolation=cv2.INTER_LINEAR)
                        dataset_base = tactile_streams[sensor_idx]
                        replay_buffer.data[f"{dataset_base}_rgb"][buffer_start + offset] = sensor_frame

                        if write_markers and sensor_kpts[sensor_idx] is not None:
                            marker_flow = sensor_kpts[sensor_idx]
                            if frame_idx < len(marker_flow):
                                replay_buffer.data[f"{dataset_base}_marker"][buffer_start + offset] = marker_flow[frame_idx]
        finally:
            camera_cap.release()
            for cap in sensor_caps.values():
                cap.release()

        click.echo(f"[dataset] wrote frames from {demo_dir.name}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zarr.ZipStore(str(output), mode="w") as zip_store:
        replay_buffer.save_to_store(store=zip_store)
    click.echo(f"[dataset] saved {output}")


if __name__ == "__main__":
    main()
