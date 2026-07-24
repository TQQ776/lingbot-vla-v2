from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCHER = ROOT / "scripts/train_tacthru_umi_v2.sh"
TACTILE_LAUNCHER = ROOT / "scripts/train_tacthru_umi_v2_tactile.sh"


def _contract_finalizer_source() -> str:
    source = BASE_LAUNCHER.read_text(encoding="utf-8")
    anchor = source.index("# The tactile launcher creates this manifest")
    start = source.index("<<'PY'\n", anchor) + len("<<'PY'\n")
    end = source.index("\nPY\nfi", start)
    return source[start:end]


@pytest.mark.parametrize(
    "override",
    [
        "--train.output_dir=/tmp/contract-bypass",
        "--train.tactile_marker_enabled=false",
        "--data.train_path=/tmp/other-dataset",
    ],
)
def test_base_launcher_rejects_equals_form_protected_overrides(override: str) -> None:
    env = {
        **os.environ,
        "DATASET_TACTILE_MODE": "rgb-marker",
        "TRAIN_TACTILE_MODE": "rgb-marker",
    }
    result = subprocess.run(
        ["bash", str(BASE_LAUNCHER), "/path/need-not-exist", "--", override],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Protected tactile launcher option cannot be overridden" in result.stderr


def test_tactile_launcher_rejects_override_before_creating_output(tmp_path: Path) -> None:
    dataset = tmp_path / "source.zarr.zip"
    dataset.touch()
    output = tmp_path / "must-remain-absent"
    result = subprocess.run(
        [
            "bash",
            str(TACTILE_LAUNCHER),
            str(dataset),
            "--output-dir",
            str(output),
            "--",
            "--train.tactile_params={}",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Protected tactile launcher option cannot be overridden" in result.stderr
    assert not output.exists()


def test_finalized_experiment_contract_and_dataset_snapshot_are_immutable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    experiment = output / "tactile_experiment.json"
    dataset_manifest = tmp_path / "tacthru_umi_v2_conversion.json"
    norm = tmp_path / "norm.json"
    config = tmp_path / "tactile.yaml"
    dataset_manifest.write_text('{"source_episodes": 2}\n', encoding="utf-8")
    norm.write_text('{"action": {}}\n', encoding="utf-8")
    config.write_text("train:\n  tactile_params: {}\n", encoding="utf-8")
    experiment.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_tactile_mode": "rgb-marker",
                "train_tactile_mode": "marker",
                "tactile_train_stage": "full",
                "source": "/tmp/source.zarr.zip",
                "lerobot_dataset": "/tmp/lerobot",
                "norm_stats": str(norm),
                "initial_model": "/tmp/model",
                "base_model_assets": "/tmp/model",
                "train_overrides": ["--train.max_steps", "10"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-c",
        _contract_finalizer_source(),
        str(experiment),
        str(dataset_manifest),
        str(norm),
        str(config),
        str(ROOT),
    ]

    first = subprocess.run(command, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    snapshot = output / "tactile_dataset_manifest.json"
    assert snapshot.read_bytes() == dataset_manifest.read_bytes()
    finalized_experiment = experiment.read_bytes()

    repeat = subprocess.run(command, text=True, capture_output=True, check=False)
    assert repeat.returncode == 0, repeat.stderr
    assert experiment.read_bytes() == finalized_experiment

    dataset_manifest.write_text('{"source_episodes": 999}\n', encoding="utf-8")
    changed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert changed.returncode != 0
    assert "Refusing to mutate or repair a finalized tactile experiment contract" in changed.stderr
    assert experiment.read_bytes() == finalized_experiment
    assert snapshot.read_text(encoding="utf-8") == '{"source_episodes": 2}\n'
