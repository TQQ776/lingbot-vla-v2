from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_architecture_switches_default_to_false_for_old_configs():
    source = (
        ROOT / "lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    init = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LingbotVLAConfig"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    args = init.args.args
    defaults = init.args.defaults
    default_by_name = {
        arg.arg: default
        for arg, default in zip(args[-len(defaults) :], defaults)
    }
    for name in ("tactile_rgb_enabled", "tactile_marker_enabled"):
        assert isinstance(default_by_name[name], ast.Constant)
        assert default_by_name[name].value is False
    assert isinstance(default_by_name["tactile_train_stage"], ast.Constant)
    assert default_by_name["tactile_train_stage"].value == "full"


def test_tactile_modules_are_instantiated_only_inside_enabled_guard():
    source = (
        ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    flow_init = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FlowMatchingV2":
            flow_init = next(
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            )
            break
    assert flow_init is not None
    guarded_sources = [
        ast.get_source_segment(source, node) or ""
        for node in ast.walk(flow_init)
        if isinstance(node, ast.If)
        and ast.get_source_segment(source, node.test) == "self.tactile_enabled"
    ]
    assert guarded_sources
    guarded = "\n".join(guarded_sources)
    assert "self.tactile_rgb_encoder =" in guarded
    assert "self.tactile_marker_encoder =" in guarded
    assert "self.tactile_fusion =" in guarded


def test_checkpoint_loader_whitelists_only_tactile_prefixes():
    source = (ROOT / "lingbotvla/models/loader.py").read_text(encoding="utf-8")
    assert '"model.tactile_rgb_encoder."' in source
    assert '"model.tactile_marker_encoder."' in source
    assert '"model.tactile_fusion."' in source
    module_utils = (ROOT / "lingbotvla/models/module_utils.py").read_text(
        encoding="utf-8"
    )
    assert "Missing non-tactile parameters" in module_utils
    assert "Partially populated tactile checkpoint" in module_utils


def test_three_stage_tactile_finetuning_is_explicit():
    source = (
        ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    assert "def _configure_tactile_train_stage" in source
    assert 'stage == "full"' in source
    assert 'stage == "expert"' in source
    assert "self.tactile_fusion.requires_grad_(True)" in source
    assert "self.qwenvl_with_expert.qwen_expert.requires_grad_(True)" in source
