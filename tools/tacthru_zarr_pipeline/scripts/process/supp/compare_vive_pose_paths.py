from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import click
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from real_world.vive_calibration import (
    load_vive_tcp_calibration,
    tracker_relative_to_newtcp_relative,
    vive_absolute_to_newtcp_absolute,
)


def mat_to_pose6d(mats: np.ndarray) -> np.ndarray:
    pos = mats[:, :3, 3]
    rot = Rotation.from_matrix(mats[:, :3, :3]).as_rotvec()
    return np.concatenate([pos, rot], axis=-1)


def fmt_vec(vec: np.ndarray, precision: int = 4) -> str:
    return np.array2string(np.asarray(vec), precision=precision, suppress_small=True)


def sorted_dirs_by_suffix(path: Path, prefix: str) -> list[Path]:
    dirs = [p for p in path.iterdir() if p.is_dir() and p.name.startswith(prefix)]

    def key_fn(p: Path) -> tuple[int, str]:
        try:
            return (int(p.name.split("-")[-1]), p.name)
        except ValueError:
            return (10**9, p.name)

    return sorted(dirs, key=key_fn)


def resolve_vive_path(path: Path, demo_index: int | None) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name != "vive.pkl":
            raise click.ClickException(f"Expected a vive.pkl file, got: {path}")
        return path

    direct = path / "vive.pkl"
    if direct.is_file():
        return direct

    tacthru_dir = path / "tacthru_vive"
    if tacthru_dir.is_dir():
        raw_dirs = sorted_dirs_by_suffix(tacthru_dir, "test-")
        if not raw_dirs:
            raise click.ClickException(f"No test-* directories found under {tacthru_dir}")
        if demo_index is None:
            demo_index = 0
        if demo_index < 0 or demo_index >= len(raw_dirs):
            raise click.ClickException(f"demo-index {demo_index} out of range [0, {len(raw_dirs) - 1}]")
        vive_path = raw_dirs[demo_index] / "vive.pkl"
        if vive_path.is_file():
            return vive_path

    raise click.ClickException(f"Could not resolve vive.pkl from {path}")


def load_dataset_plan(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return pickle.load(f)


def summarize_path(name: str, pose6d: np.ndarray, steps: int) -> None:
    pos = pose6d[:, :3]
    deltas = np.diff(pos, axis=0)
    click.echo(f"\n[{name}]")
    click.echo(f"frames={len(pos)}")
    click.echo(f"start_pos={fmt_vec(pos[0])}")
    click.echo(f"end_pos={fmt_vec(pos[-1])}")
    click.echo(f"net_delta={fmt_vec(pos[-1] - pos[0])}")
    if len(deltas) > 0:
        click.echo(f"mean_step_delta={fmt_vec(deltas.mean(axis=0))}")
        click.echo(f"median_step_delta={fmt_vec(np.median(deltas, axis=0))}")
        click.echo(f"step_norm mean={np.linalg.norm(deltas, axis=1).mean():.6f} max={np.linalg.norm(deltas, axis=1).max():.6f}")
        click.echo("first_step_deltas:")
        for i, delta in enumerate(deltas[:steps]):
            click.echo(f"  {i:03d}: {fmt_vec(delta)}")


def compare_paths(name_a: str, pose_a: np.ndarray, name_b: str, pose_b: np.ndarray) -> None:
    n = min(len(pose_a), len(pose_b))
    if n == 0:
        return
    diff = pose_a[:n, :3] - pose_b[:n, :3]
    diff_norm = np.linalg.norm(diff, axis=1)
    click.echo(
        f"[compare] {name_a} vs {name_b}: frames={n} "
        f"mean_pos_err={diff_norm.mean():.6f} max_pos_err={diff_norm.max():.6f} "
        f"start_err={diff_norm[0]:.6f} end_err={diff_norm[-1]:.6f}"
    )


@click.command(help="Inspect Vive tracker poses and the calibrated new_tcp trajectory.")
@click.argument("input_path", type=click.Path(path_type=Path, exists=True))
@click.option("--demo-index", type=int, default=None, help="When input_path is a session/task directory, select which test-* demo to inspect.")
@click.option("--dataset-plan", type=click.Path(path_type=Path, exists=True), default=None, help="Optional dataset_plan_vive.pkl to compare against.")
@click.option("--episode-index", type=int, default=0, show_default=True, help="Episode index inside dataset_plan_vive.pkl.")
@click.option("--print-steps", type=int, default=8, show_default=True, help="How many initial per-step deltas to print.")
@click.option("--export-csv", type=click.Path(path_type=Path), default=None, help="Optional CSV path for exporting aligned position traces.")
def main(
    input_path: Path,
    demo_index: int | None,
    dataset_plan: Path | None,
    episode_index: int,
    print_steps: int,
    export_csv: Path | None,
) -> None:
    vive_path = resolve_vive_path(input_path, demo_index=demo_index)
    click.echo(f"vive_path: {vive_path}")

    with vive_path.open("rb") as f:
        data = pickle.load(f)
    tracker_raw = np.asarray(data["poses"], dtype=np.float64)
    timestamps = np.asarray(data["ts"], dtype=np.float64)
    if tracker_raw.ndim != 3 or tracker_raw.shape[1:] != (4, 4):
        raise click.ClickException(f"Unexpected pose shape: {tracker_raw.shape}")

    calibration = load_vive_tcp_calibration()
    newtcp_relative = tracker_relative_to_newtcp_relative(
        tracker_raw,
        calibration.tracker_to_newtcp,
    )

    tracker_pose6d = mat_to_pose6d(tracker_raw)
    newtcp_relative_pose6d = mat_to_pose6d(newtcp_relative)

    click.echo(f"frames={len(tracker_raw)} timestamps=[{timestamps[0]:.6f}, {timestamps[-1]:.6f}] duration={timestamps[-1] - timestamps[0]:.6f}s")
    summarize_path("tracker_raw", tracker_pose6d, steps=print_steps)
    summarize_path("newtcp_relative", newtcp_relative_pose6d, steps=print_steps)

    newtcp_absolute_pose6d = None
    if "poses_bs" in data:
        tracker_absolute = np.asarray(data["poses_bs"], dtype=np.float64)
        newtcp_absolute = vive_absolute_to_newtcp_absolute(
            tracker_absolute,
            calibration.base_to_vive,
            calibration.tracker_to_newtcp,
        )
        newtcp_absolute_pose6d = mat_to_pose6d(newtcp_absolute)
        summarize_path("newtcp_absolute_diagnostic", newtcp_absolute_pose6d, steps=print_steps)

    plan_pose6d = None
    if dataset_plan is not None:
        plan = load_dataset_plan(dataset_plan.expanduser().resolve())
        if episode_index < 0 or episode_index >= len(plan):
            raise click.ClickException(f"episode-index {episode_index} out of range [0, {len(plan) - 1}]")
        tcp_pose = np.asarray(plan[episode_index]["grippers"][0]["tcp_pose"], dtype=np.float64)
        if tcp_pose.ndim != 2 or tcp_pose.shape[1] != 6:
            raise click.ClickException(f"Unexpected tcp_pose shape in dataset plan: {tcp_pose.shape}")
        plan_pose6d = tcp_pose
        click.echo(f"\n[dataset_plan] episode_index={episode_index} frames={len(plan_pose6d)}")
        compare_paths("dataset_plan_tcp", plan_pose6d, "newtcp_relative", newtcp_relative_pose6d)
        summarize_path("dataset_plan_tcp", plan_pose6d, steps=print_steps)

    if export_csv is not None:
        export_csv = export_csv.expanduser().resolve()
        n = len(tracker_pose6d)
        frame_idx = np.arange(n, dtype=np.int64)
        out = pd.DataFrame(
            {
                "frame_idx": frame_idx,
                "timestamp": timestamps[:n],
                "tracker_x": tracker_pose6d[:, 0],
                "tracker_y": tracker_pose6d[:, 1],
                "tracker_z": tracker_pose6d[:, 2],
                "newtcp_rel_x": newtcp_relative_pose6d[:, 0],
                "newtcp_rel_y": newtcp_relative_pose6d[:, 1],
                "newtcp_rel_z": newtcp_relative_pose6d[:, 2],
            }
        )
        if newtcp_absolute_pose6d is not None:
            out["newtcp_abs_x"] = newtcp_absolute_pose6d[:n, 0]
            out["newtcp_abs_y"] = newtcp_absolute_pose6d[:n, 1]
            out["newtcp_abs_z"] = newtcp_absolute_pose6d[:n, 2]
        if plan_pose6d is not None:
            m = min(len(out), len(plan_pose6d))
            out = out.iloc[:m].copy()
            out["dataset_tcp_x"] = plan_pose6d[:m, 0]
            out["dataset_tcp_y"] = plan_pose6d[:m, 1]
            out["dataset_tcp_z"] = plan_pose6d[:m, 2]
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(export_csv, index=False)
        click.echo(f"\nexported: {export_csv}")


if __name__ == "__main__":
    main()
