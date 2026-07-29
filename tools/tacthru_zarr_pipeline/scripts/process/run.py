from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import click

from supp.pipeline_utils import (
    load_sensor_cfg,
    repo_root,
    resolve_tactile_cfg_names,
    sorted_demo_dirs,
    sorted_recording_dirs,
    tactile_config_names,
    tactile_dataset_key_map,
)


STAGES = ["tactile_pre", "sync", "tactile_track", "plan", "dataset"]


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    click.echo(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _copy_marker_refs(session_dir: Path, gripper: str, swap_lr: bool) -> None:
    recording_dirs = sorted_recording_dirs(session_dir)
    if recording_dirs:
        cfg_names = resolve_tactile_cfg_names(recording_dirs[0], gripper=gripper, swap_lr=swap_lr)
    else:
        cfg_names = tactile_config_names(gripper=gripper, swap_lr=swap_lr)
    dataset_key_map = tactile_dataset_key_map(swap_lr=swap_lr)

    for sensor_idx, cfg_name in enumerate(cfg_names):
        sensor_cfg = load_sensor_cfg(cfg_name)
        tracking_rel_path = sensor_cfg["tracking"]["tracking_pts_path"]
        src_path = (repo_root() / tracking_rel_path).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing tactile marker reference file: {src_path}")
        dst_path = session_dir / f"{dataset_key_map[sensor_idx]}_marker_ref.npy"
        shutil.copy2(src_path, dst_path)


def _maybe_prepend_prepare_stage(session_dir: Path, stages_to_run: list[str]) -> list[str]:
    """当用户直接从 sync 起跑时，自动补齐 demos 准备阶段。"""
    if "tactile_pre" in stages_to_run:
        return stages_to_run
    if "sync" not in stages_to_run:
        return stages_to_run
    if sorted_demo_dirs(session_dir):
        return stages_to_run
    if not sorted_recording_dirs(session_dir):
        return stages_to_run

    click.echo("[run] demos/ missing; automatically running tactile_pre before sync for test-* recordings")
    return ["tactile_pre", *stages_to_run]


@click.command(help="Run the Demo processing pipeline end-to-end or between selected stages.")
@click.argument("session_dir", type=click.Path(path_type=Path, exists=True))
@click.option("--gripper", type=click.Choice(["a", "b", "m"]), required=True, help="Tactile sensor configuration family.")
@click.option("--swap-lr/--no-swap-lr", default=False, help="Swap the tactile sensor configs assigned to TacThru-0/1.")
@click.option("--from-stage", "from_stage", type=click.Choice(STAGES), default=STAGES[0], show_default=True)
@click.option("--to-stage", type=click.Choice(STAGES), default=STAGES[-1], show_default=True)
@click.option("--out-fov", type=float, default=120.0, show_default=True)
@click.option(
    "--camera-intrinsics",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional fisheye intrinsics JSON; omitted for the Synria C10.",
)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--num-workers", type=int, default=None)
@click.option("--force", is_flag=True, default=False)
def main(
    session_dir: Path,
    gripper: str,
    swap_lr: bool,
    from_stage: str,
    to_stage: str,
    out_fov: float,
    camera_intrinsics: Path | None,
    output: Path | None,
    num_workers: int | None,
    force: bool,
) -> None:
    repo = repo_root()
    session_dir = session_dir.expanduser().resolve()
    output = output.expanduser().resolve() if output is not None else session_dir / f"{session_dir.name}.zarr.zip"

    start_idx = STAGES.index(from_stage)
    end_idx = STAGES.index(to_stage)
    if start_idx > end_idx:
        raise click.ClickException("--from-stage must not be after --to-stage.")

    process_dir = repo / "scripts" / "process"
    stages_to_run = list(STAGES[start_idx : end_idx + 1])
    stages_to_run = _maybe_prepend_prepare_stage(session_dir=session_dir, stages_to_run=stages_to_run)
    python = sys.executable

    _copy_marker_refs(session_dir=session_dir, gripper=gripper, swap_lr=swap_lr)

    for stage in stages_to_run:
        if stage == "tactile_pre":
            cmd = [
                python,
                str(process_dir / "0_process_tacthru.py"),
                str(session_dir),
                "--mode",
                "prepare",
                "--gripper",
                gripper,
            ]
            if swap_lr:
                cmd.append("--swap-lr")
            if force:
                cmd.append("--force")
        elif stage == "sync":
            cmd = [python, str(process_dir / "1_process_videos.py"), str(session_dir)]
            if force:
                cmd.append("--force")
        elif stage == "tactile_track":
            cmd = [
                python,
                str(process_dir / "0_process_tacthru.py"),
                str(session_dir),
                "--mode",
                "track",
                "--gripper",
                gripper,
            ]
            if swap_lr:
                cmd.append("--swap-lr")
            if force:
                cmd.append("--force")
        elif stage == "plan":
            cmd = [
                python,
                str(process_dir / "3_gen_ds_plan.py"),
                "--input",
                str(session_dir),
            ]
            if force:
                cmd.append("--force")
        elif stage == "dataset":
            cmd = [
                python,
                str(process_dir / "4_gen_ds.py"),
                str(session_dir),
                "--output",
                str(output),
                "--out-fov",
                str(out_fov),
            ]
            if camera_intrinsics is not None:
                cmd.extend(["--camera-intrinsics", str(camera_intrinsics.expanduser().resolve())])
            if swap_lr:
                cmd.append("--swap-lr")
            if num_workers is not None:
                cmd.extend(["--num-workers", str(num_workers)])
            if force:
                cmd.append("--force")
        else:
            raise RuntimeError(f"Unsupported stage: {stage}")

        _run_cmd(cmd=cmd, cwd=repo)


if __name__ == "__main__":
    main()
