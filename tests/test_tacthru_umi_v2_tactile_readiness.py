from pathlib import Path

from scripts.check_tacthru_umi_v2_ready import (
    Report,
    _tactile_modalities,
    check_tactile_config,
)


ROOT = Path(__file__).resolve().parents[1]
TACTILE_CONFIG = ROOT / "configs/vla/tacthru_umi/tacthru_umi_tactile.yaml"


def test_dataset_superset_can_serve_each_architecture_ablation() -> None:
    dataset_modalities = _tactile_modalities("rgb-marker")
    for mode in ("none", "rgb", "marker", "rgb-marker"):
        assert _tactile_modalities(mode).issubset(dataset_modalities)


def test_marker_only_readiness_does_not_require_rgb_backbone_file() -> None:
    report = Report()
    check_tactile_config(report, TACTILE_CONFIG, "marker")
    assert not report.failed
    assert any(
        check["name"] == "Tactile RGB backbone"
        and "not required" in check["detail"]
        for check in report.checks
    )


def test_rgb_readiness_requires_local_nonempty_backbone() -> None:
    report = Report()
    check_tactile_config(report, TACTILE_CONFIG, "rgb")
    backbone_checks = [
        check for check in report.checks if check["name"] == "Tactile RGB backbone"
    ]
    assert backbone_checks
    configured = ROOT / "models/dinov2_vits14_pretrain.pth"
    if configured.is_file() and configured.stat().st_size > 0:
        assert backbone_checks[-1]["status"] == "ok"
        assert "sha256=" in backbone_checks[-1]["detail"]
    else:
        assert backbone_checks[-1]["status"] == "failed"
