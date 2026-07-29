from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import click

from supp.pipeline_utils import sorted_demo_dirs, write_gripper_interp_csv, write_vive_interp_csv
from supp.sync_video import synchronize_videos


def _sync_demo_dir(demo_dir: Path, tactile_latency: float, vive_latency: float, force: bool) -> None:
    raw_video = demo_dir / "raw_video.mp4"
    raw_video_ts = demo_dir / "raw_video.txt"
    vive_pkl = demo_dir / "vive.pkl"
    vive_interp_csv = demo_dir / "vive_interp.csv"
    gripper_pkl = demo_dir / "gripper.pkl"
    gripper_interp_csv = demo_dir / "gripper_interp.csv"
    tactile_indices = [
        sensor_idx
        for sensor_idx in (0, 1)
        if (demo_dir / f"TacThru-{sensor_idx}.avi").is_file()
        and (demo_dir / f"TacThru-{sensor_idx}.txt").is_file()
    ]
    if not tactile_indices:
        raise FileNotFoundError(f"Missing tactile inputs for {demo_dir}: expected at least one TacThru-*.avi/.txt pair")

    required_inputs = [
        raw_video,
        raw_video_ts,
        vive_pkl,
        gripper_pkl,
    ]
    for sensor_idx in tactile_indices:
        required_inputs.extend([demo_dir / f"TacThru-{sensor_idx}.avi", demo_dir / f"TacThru-{sensor_idx}.txt"])
    missing = [path for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs for {demo_dir}: {', '.join(str(path.name) for path in missing)}")

    write_vive_interp_csv(vive_pkl_path=vive_pkl, output_csv_path=vive_interp_csv, force=force)
    write_gripper_interp_csv(
        gripper_pkl_path=gripper_pkl,
        output_csv_path=gripper_interp_csv,
        force=force,
    )

    synced_outputs = [
        demo_dir / "camera_synced.mp4",
        demo_dir / "vive_synced.csv",
        demo_dir / "gripper_synced.csv",
    ]
    synced_outputs.extend(demo_dir / f"TacThru-{sensor_idx}_synced.mp4" for sensor_idx in tactile_indices)
    if (not force) and all(path.is_file() for path in synced_outputs):
        click.echo(f"[sync] {demo_dir.name}: already synchronized")
        return

    click.echo(f"[sync] {demo_dir.name}")
    synchronize_videos(str(demo_dir), latency=tactile_latency, vive_latency=vive_latency, do_undistort=False)


@click.command(help="Convert Vive/gripper feedback to CSV and synchronize camera, tactile, Vive, and gripper streams.")
@click.argument("session_dir", nargs=-1, required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--tactile-latency", type=float, default=-0.15, show_default=True)
@click.option("--vive-latency", type=float, default=0.0, show_default=True)
@click.option("--force", is_flag=True, default=False)
def main(session_dir: tuple[Path, ...], tactile_latency: float, vive_latency: float, force: bool) -> None:
    for session in session_dir:
        demo_dirs = sorted_demo_dirs(session)
        if not demo_dirs:
            raise click.ClickException(f"No demo directories found under {session / 'demos'}")
        for demo_dir in demo_dirs:
            _sync_demo_dir(demo_dir=demo_dir, tactile_latency=tactile_latency, vive_latency=vive_latency, force=force)


if __name__ == "__main__":
    main()
