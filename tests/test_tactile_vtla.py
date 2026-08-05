from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import einops
import pytest
import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_tactile_model = _load_module(
    "_test_tactile_vtla_module",
    ROOT / "lingbotvla/models/vla/lingbot_vla/tactile_vtla.py",
)
_tactile_data = _load_module(
    "_test_tactile_data_module",
    ROOT / "lingbotvla/data/vla_data/tactile.py",
)
_marker_stats = _load_module(
    "_test_tactile_marker_stats_module",
    ROOT / "tools/compute_tacthru_marker_stats.py",
)
_vtla_optim = _load_module(
    "_test_tactile_optimizer_module",
    ROOT / "lingbotvla/optim/vtla.py",
)

MarkerEncoder = _tactile_model.MarkerEncoder
TactileTokenEncoder = _tactile_model.TactileTokenEncoder
TactileVTLAConfig = _tactile_model.TactileVTLAConfig
concatenate_vtla_context = _tactile_model.concatenate_vtla_context
marker_features_from_positions = _tactile_model.marker_features_from_positions
prepare_tactile_sample = _tactile_data.prepare_tactile_sample


def _settings(**overrides) -> TactileVTLAConfig:
    values = {
        "enabled": True,
        "num_sensors": 2,
        "num_markers": 4,
        "use_rgb": True,
        "use_markers": True,
        "marker_hidden_dim": 16,
        "marker_mean": [0.0, 0.0, 0.0, 0.0],
        "marker_std": [1.0, 1.0, 1.0, 1.0],
        "rgb_keys": ["rgb_l", "rgb_r"],
        "marker_positions_keys": ["pos_l", "pos_r"],
        "marker_reference_keys": ["ref_l", "ref_r"],
        "marker_valid_mask_keys": ["valid_l", "valid_r"],
        "marker_flow_keys": ["flow_l", "flow_r"],
    }
    values.update(overrides)
    return TactileVTLAConfig.from_mapping(values)


def test_config_defaults_to_tactile_disabled_and_rejects_invalid_modes():
    config = TactileVTLAConfig.from_mapping(None)
    assert config.enabled is False

    with pytest.raises(ValueError, match="requires use_rgb or use_markers"):
        TactileVTLAConfig.from_mapping(
            {"enabled": True, "use_rgb": False, "use_markers": False}
        )
    with pytest.raises(ValueError, match="num_sensors"):
        TactileVTLAConfig.from_mapping({"enabled": True, "num_sensors": 0})
    with pytest.raises(ValueError, match="marker_input_features"):
        TactileVTLAConfig.from_mapping(
            {"enabled": True, "marker_input_features": 6}
        )


def test_saved_marker_stats_remain_portable_without_original_path(tmp_path):
    settings = _settings(
        marker_stats_path=str(tmp_path / "server-only-stats.json"),
        marker_mean=[1.0, 2.0, 3.0, 4.0],
        marker_std=[0.5, 0.6, 0.7, 0.8],
    )
    assert settings.marker_mean == (1.0, 2.0, 3.0, 4.0)
    assert settings.marker_std == pytest.approx((0.5, 0.6, 0.7, 0.8))


def test_marker_features_are_displacement_and_velocity_with_mask():
    positions = torch.tensor([[[[3.0, 5.0], [7.0, 11.0]]]])
    reference = torch.tensor([[[[1.0, 2.0], [3.0, 5.0]]]])
    previous = torch.tensor([[[[2.0, 4.0], [6.0, 9.0]]]])
    valid = torch.tensor([[[True, False]]])
    features, output_mask = marker_features_from_positions(
        positions, reference, previous, valid
    )
    assert output_mask.equal(valid)
    assert features[0, 0, 0].tolist() == [2.0, 3.0, 1.0, 1.0]
    assert torch.count_nonzero(features[0, 0, 1]) == 0


def test_marker_encoder_single_and_two_sensor_shapes():
    encoder = MarkerEncoder(
        num_markers=4,
        context_dim=8,
        hidden_dim=16,
    )
    single = encoder(torch.randn(3, 4, 4), torch.ones(3, 4, dtype=torch.bool))
    dual = encoder(
        torch.randn(3, 2, 4, 4),
        torch.ones(3, 2, 4, dtype=torch.bool),
    )
    assert single.shape == (3, 1, 8)
    assert dual.shape == (3, 2, 8)


def test_marker_encoder_masks_fully_missing_sensor_after_mlp_bias():
    encoder = MarkerEncoder(num_markers=4, context_dim=8, hidden_dim=16)
    features = torch.randn(2, 2, 4, 4)
    valid = torch.ones(2, 2, 4, dtype=torch.bool)
    valid[:, 1] = False
    tokens = encoder(features, valid)
    assert tokens.shape == (2, 2, 8)
    assert torch.count_nonzero(tokens[:, 1]) == 0


def test_tactile_rgb_projection_shape_and_missing_sensor_mask():
    encoder = TactileTokenEncoder(
        _settings(use_markers=False),
        context_dim=8,
        vision_output_dim=6,
    )
    embeddings = torch.randn(2, 2, 5, 6)
    tokens, mask = encoder.encode_rgb_embeddings(
        embeddings,
        tactile_sensor_mask=torch.tensor([[True, False], [True, True]]),
    )
    assert tokens.shape == (2, 10, 8)
    assert mask.shape == (2, 10)
    assert not mask[0, 5:].any()
    assert torch.count_nonzero(tokens[0, 5:]) == 0


def test_vtla_context_and_mask_lengths_are_extended_on_token_axis():
    vl = torch.randn(2, 7, 8)
    vl_mask = torch.ones(2, 7, dtype=torch.bool)
    rgb = torch.randn(2, 10, 8)
    rgb_mask = torch.ones(2, 10, dtype=torch.bool)
    marker = torch.randn(2, 2, 8)
    marker_mask = torch.ones(2, 2, dtype=torch.bool)
    context, mask = concatenate_vtla_context(
        vl, vl_mask, [(rgb, rgb_mask), (marker, marker_mask)]
    )
    assert context.shape == (2, 19, 8)
    assert mask.shape == (2, 19)


def test_tactile_disabled_regression_adds_no_context_tokens():
    vl = torch.randn(2, 7, 8)
    vl_mask = torch.ones(2, 7, dtype=torch.bool)
    context, mask = concatenate_vtla_context(vl, vl_mask, [])
    assert torch.equal(context, vl)
    assert torch.equal(mask, vl_mask)


def test_backward_reaches_new_modules_and_action_but_not_frozen_vision():
    torch.manual_seed(0)
    vision = nn.Linear(3, 6, bias=False).requires_grad_(False)
    tactile = TactileTokenEncoder(
        _settings(),
        context_dim=8,
        vision_output_dim=6,
    )
    action_expert = nn.Linear(8, 4)

    rgb_input = torch.randn(2, 2, 5, 3)
    rgb_emb = vision(rgb_input)
    rgb_tokens, _ = tactile.encode_rgb_embeddings(
        rgb_emb,
        tactile_sensor_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    positions = torch.randn(2, 2, 4, 2)
    marker_tokens, _ = tactile.encode_markers(
        positions,
        torch.zeros_like(positions),
        positions - 0.1,
        torch.ones(2, 2, 4, dtype=torch.bool),
        torch.ones(2, 2, dtype=torch.bool),
    )
    prediction = action_expert(torch.cat([rgb_tokens, marker_tokens], dim=1).mean(1))
    prediction.square().mean().backward()

    assert vision.weight.grad is None
    assert tactile.tactile_rgb_projection.weight.grad is not None
    assert tactile.marker_encoder.encoder[0].weight.grad is not None
    assert tactile.tactile_rgb_modality_embedding.grad is not None
    assert tactile.marker_modality_embedding.grad is not None
    assert tactile.sensor_side_embeddings.weight.grad is not None
    assert action_expert.weight.grad is not None


def test_optimizer_groups_are_disjoint_and_exclude_frozen_parameters():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tactile_encoder = nn.Linear(2, 2)
            self.qwen_expert = nn.Linear(2, 2)
            self.depth_head = nn.Linear(2, 2)
            self.frozen_visual = nn.Linear(2, 2).requires_grad_(False)

    model = TinyModel()
    groups = _vtla_optim.build_vtla_param_groups(
        model,
        new_modules_lr=1e-4,
        action_expert_lr=1e-5,
        base_lr=2e-5,
        weight_decay=1e-2,
    )
    by_name = {group["name"]: group for group in groups}
    assert by_name["tactile"]["lr"] == pytest.approx(1e-4)
    assert by_name["action_expert"]["lr"] == pytest.approx(1e-5)
    assert by_name["base_trainable"]["lr"] == pytest.approx(2e-5)
    identifiers = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(identifiers) == len(set(identifiers))
    assert id(model.frozen_visual.weight) not in identifiers
    assert id(model.frozen_visual.bias) not in identifiers


class _FakeImageProcessor:
    merge_size = 2

    def __call__(self, image):
        # Preserve deterministic values while mimicking Qwen's flattened patch
        # representation and grid contract.
        value = image.to(torch.float32).mean().reshape(1, 1).expand(4, 6).clone()
        return {
            "pixel_values": value,
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


def test_data_transform_uses_flow_as_equivalent_position_contract():
    settings = _settings()
    flow_l = torch.stack(
        [torch.zeros(4, 2), torch.full((4, 2), 0.25)], dim=0
    )
    sample = {
        "flow_l": flow_l,
        "valid_l": torch.ones(2, 4, dtype=torch.bool),
        "rgb_l": torch.ones(3, 8, 8, dtype=torch.uint8),
        "scene": torch.zeros(3, 8, 8, dtype=torch.uint8),
    }
    output = prepare_tactile_sample(
        sample,
        _FakeImageProcessor(),
        settings,
        fallback_image_keys=("scene",),
    )
    assert output["marker_positions"].shape == (2, 4, 2)
    assert torch.equal(output["marker_positions"][0], flow_l[-1])
    assert torch.equal(output["marker_reference"][0], torch.zeros_like(flow_l[-1]))
    assert torch.equal(output["previous_marker_positions"][0], flow_l[-2])
    assert output["tactile_rgb"].shape == (2, 4, 6)
    assert output["tactile_rgb_mask"].tolist() == [True, False]
    assert output["tactile_sensor_mask"].tolist() == [True, False]


def test_inference_data_accepts_aggregate_tactile_fields_and_sensor_mask():
    settings = _settings()
    positions = torch.randn(2, 4, 2)
    previous = positions - 0.2
    sample = {
        "marker_positions": positions,
        "marker_reference": torch.zeros_like(positions),
        "previous_marker_positions": previous,
        "marker_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        "tactile_rgb": torch.ones(2, 3, 8, 8, dtype=torch.uint8),
        "tactile_sensor_mask": torch.tensor([True, False]),
        "scene": torch.zeros(3, 8, 8, dtype=torch.uint8),
    }
    output = prepare_tactile_sample(
        sample,
        _FakeImageProcessor(),
        settings,
        fallback_image_keys=("scene",),
    )
    assert torch.equal(output["previous_marker_positions"], previous)
    assert not output["marker_valid_mask"][1].any()
    assert output["tactile_rgb_mask"].tolist() == [True, False]
    assert output["tactile_sensor_mask"].tolist() == [True, False]


def test_marker_stats_use_selected_episodes_and_reset_velocity_at_boundary():
    marker = np.asarray(
        [
            [[-100.0, -200.0]],
            [[-90.0, -180.0]],
            [[10.0, 20.0]],
            [[14.0, 28.0]],
        ],
        dtype=np.float32,
    )
    mean, std, count, frame_count = _marker_stats.compute_stats(
        marker,
        np.asarray([2, 4], dtype=np.int64),
        episode_start=1,
        episode_end=2,
        chunk_frames=1,
    )
    np.testing.assert_allclose(mean, [12.0, 24.0, 2.0, 4.0])
    np.testing.assert_allclose(std, [2.0, 4.0, 2.0, 4.0])
    assert count == 2
    assert frame_count == 2


def test_model_training_and_sampling_interfaces_expose_all_tactile_fields():
    source = (
        Path(__file__).parents[1]
        / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    signatures = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                signatures[(node.name, child.name)] = {
                    argument.arg for argument in child.args.args
                }
    required = {
        "tactile_rgb",
        "tactile_rgb_grid_thw",
        "marker_positions",
        "marker_reference",
        "previous_marker_positions",
        "marker_valid_mask",
        "tactile_sensor_mask",
        "tactile_rgb_mask",
    }
    assert required <= signatures[("FlowMatchingV2", "forward")]
    assert required <= signatures[("FlowMatchingV2", "sample_actions")]
    assert required <= signatures[("LingbotVlaV2Policy", "forward")]
    assert required <= signatures[("LingbotVlaV2Policy", "sample_actions")]


def test_lerobot_v3_shared_parquet_paths_are_deduplicated():
    source = (ROOT / "lingbotvla/data/vla_data/base_dataset.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LeRobotDataset"
    )
    method = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_hf_dataset"
    )
    assignments = {
        target.id: value
        for statement in method.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        for value in [statement.value]
    }
    files_expression = assignments["files"]
    assert isinstance(files_expression, ast.Call)
    assert isinstance(files_expression.func, ast.Name)
    assert files_expression.func.id == "sorted"
    assert isinstance(files_expression.args[0], ast.SetComp)


def _load_prefix_harness():
    """Execute the real prefix method without importing the 6B model stack."""

    source = (
        ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    flow_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMatchingV2"
    )
    method = next(
        node
        for node in flow_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_embed_prefix_vtla"
    )
    harness = ast.ClassDef(
        name="PrefixHarness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {
        "Tensor": torch.Tensor,
        "einops": einops,
        "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
        "torch": torch,
    }
    exec(compile(module, str(ROOT / "prefix_harness.py"), "exec"), namespace)
    return namespace["PrefixHarness"]


class _FakeQwenPrefixBackend:
    def __init__(self, hidden_dim: int, num_patches: int) -> None:
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.position_call = None
        self.qwenvl = SimpleNamespace(
            config=SimpleNamespace(
                image_token_id=10,
                vision_start_token_id=11,
                vision_end_token_id=12,
                text_config=SimpleNamespace(eos_token_id=13),
            )
        )

    def embed_image(self, images, image_grid_thw):
        num_views = images.shape[0]
        view_values = torch.arange(
            1,
            num_views + 1,
            dtype=images.dtype,
            device=images.device,
        ).view(num_views, 1, 1)
        embeddings = view_values.expand(
            num_views, self.num_patches, self.hidden_dim
        ).clone()
        return embeddings, [embeddings + 100]

    def embed_special_token(self, token_id, batch, count, device, dtype):
        return torch.full(
            (batch, count, 1, self.hidden_dim),
            float(token_id),
            dtype=dtype,
            device=device,
        )

    def embed_language_tokens(self, tokens):
        return tokens.to(torch.float32).unsqueeze(-1).expand(
            *tokens.shape, self.hidden_dim
        )

    def build_prefix_position_ids(
        self,
        input_ids,
        attention_mask,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        self.position_call = {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "image_grid_thw": image_grid_thw.clone(),
        }
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)


def test_full_vtla_prefix_path_preserves_order_masks_rope_and_deepstack():
    harness = _load_prefix_harness()()
    harness.tactile_settings = _settings()
    harness.tactile_encoder = TactileTokenEncoder(
        harness.tactile_settings,
        context_dim=8,
        vision_output_dim=8,
    )
    harness.qwenvl_with_expert = _FakeQwenPrefixBackend(8, 3)
    harness.config = SimpleNamespace(
        qwen3vl_use_vision_boundaries=True,
        vlm_causal=False,
    )
    harness.use_depth_align = False
    harness._tactile_debug_logged = False

    batch_size, scene_views, tactile_sensors = 2, 2, 2
    images = torch.zeros(batch_size, scene_views, 4, 6)
    tactile_rgb = torch.ones(batch_size, tactile_sensors, 4, 6)
    scene_grid = torch.ones(batch_size, scene_views, 3, dtype=torch.long)
    tactile_grid = torch.ones(
        batch_size, tactile_sensors, 3, dtype=torch.long
    )
    image_mask = torch.tensor([[True, False], [True, True]])
    sensor_mask = torch.tensor([[True, False], [True, True]])
    language = torch.tensor([[21, 22, 23, 24], [31, 32, 33, 34]])
    language_mask = torch.ones_like(language, dtype=torch.bool)
    marker_positions = torch.ones(batch_size, tactile_sensors, 4, 2)
    marker_valid = torch.ones(batch_size, tactile_sensors, 4, dtype=torch.bool)

    (
        context,
        pad_mask,
        attention_mask,
        position_ids,
        visual_mask,
        deepstack,
    ) = harness._embed_prefix_vtla(
        images=images,
        img_masks=image_mask,
        lang_tokens=language,
        lang_masks=language_mask,
        image_grid_thw=scene_grid,
        tactile_rgb=tactile_rgb,
        tactile_rgb_grid_thw=tactile_grid,
        marker_positions=marker_positions,
        marker_reference=torch.zeros_like(marker_positions),
        previous_marker_positions=marker_positions - 0.25,
        marker_valid_mask=marker_valid,
        tactile_sensor_mask=sensor_mask,
        tactile_rgb_mask=sensor_mask,
    )

    scene_length = scene_views * (3 + 2)
    language_length = language.shape[1]
    rgb_length = tactile_sensors * (3 + 2)
    marker_length = tactile_sensors
    expected_length = scene_length + language_length + rgb_length + marker_length
    assert context.shape == (batch_size, expected_length, 8)
    assert pad_mask.shape == (batch_size, expected_length)
    assert attention_mask.shape == (batch_size, expected_length)
    assert position_ids.shape == (3, batch_size, expected_length)
    assert visual_mask.shape == (batch_size, expected_length)

    # Prefix order is scene, language, tactile RGB, marker.  A missing tactile
    # sensor retains its five-token slot, but both embeddings and mask are zero.
    rgb_start = scene_length + language_length
    missing_rgb = slice(rgb_start + 5, rgb_start + 10)
    assert not pad_mask[0, missing_rgb].any()
    assert torch.count_nonzero(context[0, missing_rgb]) == 0
    marker_start = rgb_start + rgb_length
    assert not pad_mask[0, marker_start + 1]
    assert torch.count_nonzero(context[0, marker_start + 1]) == 0

    position_call = harness.qwenvl_with_expert.position_call
    assert position_call is not None
    assert position_call["input_ids"][0, scene_length:rgb_start].equal(language[0])
    assert position_call["input_ids"][0, marker_start:].tolist() == [13, 13]
    expected_valid_views = int(image_mask.sum() + sensor_mask.sum())
    assert position_call["image_grid_thw"].shape == (expected_valid_views, 3)
    expected_visual_tokens = expected_valid_views * 3
    assert int(visual_mask.sum()) == expected_visual_tokens
    assert deepstack[0].shape == (expected_visual_tokens, 8)


def test_flow_matching_and_euler_math_remain_the_original_expressions():
    source = (
        ROOT / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    flow_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMatchingV2"
    )

    def method_source(name: str) -> str:
        method = next(
            node
            for node in flow_class.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        return "\n".join(
            ast.unparse(statement) for statement in method.body
        )

    training = method_source("forward")
    sampling = method_source("sample_actions")
    assert "noise = torch.randn(actions.shape, device=device, dtype=dtype)" in training
    assert "time = self.sample_time(actions.size(0), device).to(dtype)" in training
    assert "x_t = time_expanded * noise + (1 - time_expanded) * actions" in training
    assert "u_t = noise - actions" in training
    assert "F.mse_loss(u_t, v_t, reduction='none')" in training
    assert "dt = torch.tensor(-1.0 / self.config.num_steps" in sampling
    assert "x_t += dt * v_t" in sampling
    assert "time += dt" in sampling


def test_checkpoint_compatibility_is_limited_to_tactile_namespace():
    method = TactileVTLAConfig.from_mapping(
        {"enabled": True, "num_sensors": 1, "use_rgb": False}
    )
    assert method.enabled
    model_source = (
        Path(__file__).parents[1]
        / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    assert 'tactile_prefix = "model.tactile_encoder."' in model_source
    assert "loaded_parameter_names" in model_source


def test_missing_direct_tactile_parameter_uses_leaf_initializer():
    source = (ROOT / "lingbotvla/models/module_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_init_parameter"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"nn": nn}
    exec(compile(module, str(ROOT / "module_utils_harness.py"), "exec"), namespace)

    class DirectParameter(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Parameter(torch.full((2,), float("nan")))
            self.initialized = []

        def _init_missing_parameter(self, name):
            self.initialized.append(name)
            with torch.no_grad():
                self._parameters[name].zero_()

    class RootModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.tactile_encoder = DirectParameter()

        @staticmethod
        def _init_weights(module):
            return None

    root = RootModule()
    namespace["_init_parameter"](root, "tactile_encoder.embedding")
    assert root.tactile_encoder.initialized == ["embedding"]
    assert torch.equal(root.tactile_encoder.embedding, torch.zeros(2))


@pytest.mark.parametrize(
    ("name", "enabled", "use_rgb", "use_markers"),
    [
        ("tacthru_umi_vtla_base.yaml", False, True, True),
        ("tacthru_umi_vtla_rgb.yaml", True, True, False),
        ("tacthru_umi_vtla_marker.yaml", True, False, True),
        ("tacthru_umi_vtla_rgb_marker.yaml", True, True, True),
    ],
)
def test_ablation_configs_validate(name, enabled, use_rgb, use_markers):
    path = ROOT / "configs/vla/tacthru_umi" / name
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = TactileVTLAConfig.from_mapping(payload["train"]["tactile"])
    assert settings.enabled is enabled
    assert settings.use_rgb is use_rgb
    assert settings.use_markers is use_markers
    assert payload["data"]["train_path"].endswith(
        "insert_ethernet_cable_ml_0721_201_tacthru_umi_v2_tactile_v1"
    )
    if enabled:
        assert payload["train"]["optimizer"] == "adamw"
        assert payload["train"]["freeze_vlm"] is True
