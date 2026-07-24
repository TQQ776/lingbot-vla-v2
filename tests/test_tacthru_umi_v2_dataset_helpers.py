import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


# Avoid importing the production ``lingbotvla.data.__init__`` and its optional
# training dependencies.  These tests exercise only vla_data helper modules.
repo_root = Path(__file__).resolve().parents[1]
data_package = types.ModuleType("lingbotvla.data")
data_package.__path__ = [str(repo_root / "lingbotvla" / "data")]
vla_data_package = types.ModuleType("lingbotvla.data.vla_data")
vla_data_package.__path__ = [str(repo_root / "lingbotvla" / "data" / "vla_data")]
helper_module = types.ModuleType("lingbotvla.utils.helper")
helper_module.create_logger = lambda _name: SimpleNamespace(info=lambda *_args, **_kwargs: None)
sys.modules.setdefault("lingbotvla.data", data_package)
sys.modules.setdefault("lingbotvla.data.vla_data", vla_data_package)
sys.modules.setdefault("lingbotvla.utils.helper", helper_module)
sys.modules.setdefault("einops", types.ModuleType("einops"))

from lingbotvla.data.vla_data import base_dataset as base_dataset_module  # noqa: E402
from lingbotvla.data.vla_data import utils as utils_module  # noqa: E402
from lingbotvla.data.vla_data.base_dataset import (  # noqa: E402
    VLADataset,
    build_complete_chunk_anchor_indices,
    is_tacthru_umi_v2,
)
from lingbotvla.data.vla_data.multi_vla_dataset import MultiVLADataset  # noqa: E402
from lingbotvla.data.vla_data.transform import prepare_images  # noqa: E402
from lingbotvla.data.vla_data.utils import FeatureTransform  # noqa: E402

from deploy.tacthru_umi_v2.transforms import base_pose_from_episode_action, pose_mat_from_rotvec  # noqa: E402


def test_tacthru_v2_detection_accepts_data_name_or_config_path():
    assert is_tacthru_umi_v2(data_name="tacthru_umi_v2")
    assert is_tacthru_umi_v2(data_name="tacthru-umi-v2")
    assert is_tacthru_umi_v2(robot_config_path="configs/robot_configs/tacthru_umi_v2.yaml")
    assert is_tacthru_umi_v2(data_name="tacthru_umi_v2_tactile")
    assert is_tacthru_umi_v2(
        robot_config_path="configs/robot_configs/tacthru_umi_v2_tactile.yaml"
    )
    assert not is_tacthru_umi_v2(data_name="robotwin")


def test_complete_chunk_anchors_drop_each_episode_tail_independently():
    episodes = [
        {"dataset_from_index": 0, "dataset_to_index": 50},
        {"dataset_from_index": 50, "dataset_to_index": 104},
        {"dataset_from_index": 104, "dataset_to_index": 153},
    ]

    assert build_complete_chunk_anchor_indices(episodes, chunk_size=50) == [
        0,
        50,
        51,
        52,
        53,
        54,
    ]


def test_complete_chunk_anchors_accept_column_or_index_mappings():
    columnar = {
        "dataset_from_index": [0, 100],
        "dataset_to_index": [52, 154],
    }
    indexed = {
        0: {"dataset_from_index": 0, "dataset_to_index": 52},
        1: {"dataset_from_index": 100, "dataset_to_index": 154},
    }

    expected = [0, 1, 2, 100, 101, 102, 103, 104]
    assert build_complete_chunk_anchor_indices(columnar, 50) == expected
    assert build_complete_chunk_anchor_indices(indexed, 50) == expected


def test_future_video_effective_fps_survives_feature_conversion():
    transform = FeatureTransform.__new__(FeatureTransform)
    transform.key_mapping = {}
    transform.feature_to_keep = {"future_video_effective_fps"}
    transform.disabled_image_features = True
    transform.images = []
    transform.actions_convert_from_state = {}

    value = torch.tensor(30.0 / 49.0)
    output = transform.convert_features(
        {"future_video_effective_fps": value},
        w_action=True,
    )

    torch.testing.assert_close(output["future_video_effective_fps"], value)


def test_wrist_rgb_only_visual_batch_has_one_valid_view():
    wrist_key = "observation.images.camera_wrist_left"
    images, img_masks, pil_images, image_grid_thw = prepare_images(
        image_processor=None,
        observation={
            "state": torch.zeros(55),
            "image": {wrist_key: torch.zeros(3, 8, 8, dtype=torch.uint8)},
        },
        image_keys=[wrist_key],
        train=False,
        use_depth_align=False,
    )

    assert images.shape == (1, 3, 8, 8)
    assert img_masks.tolist() == [True]
    assert pil_images == []
    assert image_grid_thw is None


def test_future_video_effective_fps_reaches_final_model_batch(monkeypatch):
    transform = FeatureTransform.__new__(FeatureTransform)
    transform.org_features = {"actions": ["action"]}
    transform.actions_convert_from_state = {}
    transform.actions = []
    transform.action_subtract_state = {}
    transform.normalizer = None
    transform.return_item_befor_padding = False
    transform.model_config = SimpleNamespace(
        max_state_dim=1,
        max_action_dim=1,
        qwen3vl_use_vision_boundaries=True,
        return_image_grid_thw=False,
    )
    transform.processor = SimpleNamespace(
        image_processor=SimpleNamespace(merge_size=2),
    )
    transform.tokenizer = None
    transform.disabled_image_features = True
    transform.use_depth_align = False
    transform.use_future_image = True
    transform.convert_features = lambda item, w_action: item
    transform.pad_and_concat = lambda item, w_action: {
        "image": {},
        "future_image": {},
        "state": torch.zeros(1),
        "action": torch.zeros(2, 1),
        "action_is_pad": torch.zeros(2, dtype=torch.bool),
        "chunk_joint_mask": torch.ones(2, 1, dtype=torch.bool),
        "action_joint_mask": torch.ones(1, dtype=torch.bool),
        "state_joint_mask": torch.ones(1, dtype=torch.bool),
        "prompt": ["Pull the tissue"],
        "future_video_effective_fps": item["future_video_effective_fps"],
    }
    monkeypatch.setattr(
        utils_module,
        "prepare_language",
        lambda *_args, **_kwargs: (
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
        ),
    )

    value = torch.tensor(30.0 / 49.0)
    output = transform.apply(
        {
            "action_is_pad": torch.zeros(2, dtype=torch.bool),
            "future_video_effective_fps": value,
        }
    )

    torch.testing.assert_close(output["future_video_effective_fps"], value)


def test_vla_dataset_uses_norm_override_and_filtered_logical_indices(monkeypatch):
    captured = {}

    class _FeatureTransform:
        def __init__(self, *_args, **kwargs):
            captured["norm_stats_path"] = kwargs["norm_stats_path"]
            self.actions = ["action.end.position"]
            self.states = ["observation.state.end.position"]
            self.images = []
            self.actions_convert_from_state = {}
            self.org_features = {"actions": [], "states": [], "images": []}
            self.feature_to_keep = set()

    class _Metadata:
        def __init__(self, repo_id, root=None):
            self.repo_id = repo_id
            self.root = root
            self.fps = 30
            self.episodes = [
                {"dataset_from_index": 0, "dataset_to_index": 50},
                {"dataset_from_index": 50, "dataset_to_index": 104},
            ]

    class _Dataset:
        def __init__(self, **_kwargs):
            pass

        def __len__(self):
            return 104

    monkeypatch.setattr(base_dataset_module, "FeatureTransform", _FeatureTransform)
    monkeypatch.setattr(base_dataset_module, "LeRobotDatasetMetadata", _Metadata)
    monkeypatch.setattr(base_dataset_module, "LeRobotDataset", _Dataset)

    dataset = VLADataset(
        repo_id="synthetic",
        data_name="tacthru_umi_v2",
        dataset_config=SimpleNamespace(norm_stats_file="custom_norm.json"),
        robot_config_root="configs/robot_configs",
        config=SimpleNamespace(),
        processor=None,
        chunk_size=50,
    )

    assert captured["norm_stats_path"] == "custom_norm.json"
    assert dataset.sample_indices == [0, 50, 51, 52, 53, 54]
    assert len(dataset) == 6


def test_tacthru_v2_future_image_uses_official_50_frame_horizon():
    dataset = VLADataset.__new__(VLADataset)
    dataset.chunk_size = 50
    dataset.use_future_image = True
    dataset.dataset_meta = SimpleNamespace(fps=30)
    dataset.feature_transform = SimpleNamespace(
        org_features={"images": ["observation.images.camera_wrist_left"]}
    )

    assert dataset.get_video_delta_timestamps() == {
        "observation.images.camera_wrist_left": [0, 49.0 / 30.0]
    }


def test_tacthru_v2_actions_use_contiguous_30_hz_frames():
    dataset = VLADataset.__new__(VLADataset)
    dataset.chunk_size = 50
    dataset.dataset_meta = SimpleNamespace(fps=30)
    dataset.feature_transform = SimpleNamespace(
        actions_convert_from_state={},
        org_features={"actions": ["action"], "states": []},
    )

    assert dataset.get_delta_timestamps()["action"] == [step / 30 for step in range(50)]


def test_tactile_history_offsets_are_independent_from_action_and_future_video():
    dataset = VLADataset.__new__(VLADataset)
    dataset.chunk_size = 50
    dataset.use_future_image = True
    dataset.dataset_meta = SimpleNamespace(fps=30)
    dataset.feature_transform = SimpleNamespace(
        actions_convert_from_state={},
        org_features={
            "actions": ["action"],
            "states": [],
            "images": ["observation.images.camera_wrist_left"],
        },
    )
    dataset.tactile_rgb_enabled = True
    dataset.tactile_marker_enabled = True
    dataset.tactile_history_steps = 4
    dataset.tactile_history_stride = 1
    dataset.tactile_rgb_key = "observation.images.tactile_left"
    dataset.tactile_marker_key = "observation.tactile.marker_flow_left"
    dataset.tactile_marker_valid_key = "observation.tactile.marker_valid_left"
    dataset.tactile_timestamp_key = "observation.tactile.timestamp"

    deltas = dataset.get_delta_timestamps()
    expected_history = [-3 / 30, -2 / 30, -1 / 30, 0.0]
    assert deltas["action"] == [step / 30 for step in range(50)]
    assert deltas[dataset.tactile_marker_key] == expected_history
    assert deltas[dataset.tactile_marker_valid_key] == expected_history
    assert deltas[dataset.tactile_timestamp_key] == expected_history

    video_deltas = dataset.get_video_delta_timestamps()
    assert video_deltas["observation.images.camera_wrist_left"] == [0, 49 / 30]
    assert video_deltas[dataset.tactile_rgb_key] == expected_history


def test_feature_transform_emits_canonical_tactile_batch_keys_without_dropout():
    transform = FeatureTransform.__new__(FeatureTransform)
    transform.tactile_rgb_enabled = True
    transform.tactile_marker_enabled = True
    transform.tactile_params = {"rgb_augment": False}
    transform.tactile_rgb_key = "observation.images.tactile_left"
    transform.tactile_marker_key = "observation.tactile.marker_flow_left"
    transform.tactile_marker_valid_key = "observation.tactile.marker_valid_left"
    transform.tactile_timestamp_key = "observation.tactile.timestamp"
    item = {
        transform.tactile_rgb_key: torch.ones(4, 3, 8, 8, dtype=torch.uint8) * 255,
        f"{transform.tactile_rgb_key}_is_pad": torch.tensor([True, True, False, False]),
        transform.tactile_marker_key: torch.ones(4, 48, 2),
        transform.tactile_marker_valid_key: torch.ones(4, 48, dtype=torch.bool),
        f"{transform.tactile_marker_key}_is_pad": torch.tensor([True, True, False, False]),
        transform.tactile_timestamp_key: torch.arange(4, dtype=torch.float64).unsqueeze(-1),
    }
    transform._tactile_train_for_current_sample = False
    payload = transform._prepare_tactile_payload(item, train=False)
    assert payload["tactile_rgb_history"].shape == (4, 3, 8, 8)
    assert payload["tactile_rgb_history_mask"].tolist() == [False, False, True, True]
    assert payload["tactile_marker_flow"].shape == (4, 48, 2)
    assert payload["tactile_marker_valid_mask"].shape == (4, 48)
    assert payload["tactile_marker_history_mask"].tolist() == [False, False, True, True]
    assert payload["tactile_history_timestamps"].shape == (4,)


class _FakeVLADataset:
    def __init__(self, name, size):
        self.data_name = name
        self.size = size
        self.dataset = SimpleNamespace(repo_id=name)

    def __len__(self):
        return self.size

    def getitem(self, idx):
        return {"source": self.data_name, "local_idx": idx}


def test_multi_dataset_indexes_filtered_logical_lengths():
    dataset = MultiVLADataset.__new__(MultiVLADataset)
    dataset._datasets = [_FakeVLADataset("first", 2), _FakeVLADataset("tacthru_umi_v2", 3)]
    dataset.dataset_start_index = [0, 2]

    assert dataset.num_frames == 5
    assert len(dataset) == 5
    assert dataset.getdata(1)["local_idx"] == 1
    second = dataset.getdata(2)
    assert second["source"] == "tacthru_umi_v2"
    assert second["local_idx"] == 0
    assert second["rep_id"] == "tacthru_umi_v2"


def test_feature_unapply_and_realman_bridge_recover_episode_absolute_pose():
    transform = FeatureTransform.__new__(FeatureTransform)
    transform.return_item_befor_padding = True
    transform.normalizer = None
    transform.actions = ["action.end.position", "action.effector.position"]
    transform.action_subtract_state = {
        "action.end.position": True,
        "action.effector.position": False,
    }
    transform.action_relative_type = {"action.end.position": "quaternion_local"}
    transform.actions_convert_from_state = {}
    transform.key_reverse_mapping = {
        "action": [
            {
                "target_key": "action.end.position",
                "target_start": 0,
                "target_end": 7,
                "end": 7,
            },
            {
                "target_key": "action.effector.position",
                "target_start": 0,
                "target_end": 1,
                "end": 8,
            },
        ]
    }
    transform.feature_to_keep = set()

    half_sqrt = np.sqrt(0.5)
    current = torch.tensor([0.4, -0.1, 0.2, 0.0, 0.0, half_sqrt, half_sqrt])
    relative = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    output = transform.unapply(
        {
            "observation.state.end.position": current,
            "action.end.position": relative,
            "action.effector.position": torch.tensor([[0.03]]),
        }
    )

    action = output["action"].numpy()[0]
    assert action[:3] == pytest.approx([0.4, 0.0, 0.2], abs=1e-6)
    assert action[3:7] == pytest.approx(current.numpy()[3:7], abs=1e-6)
    assert action[7] == pytest.approx(0.03)

    base_start = pose_mat_from_rotvec([0.5, 0.0, 0.0], [0.0, 0.0, 0.0])
    base_target = base_pose_from_episode_action(base_start, action)
    assert base_target[:3, 3] == pytest.approx([0.9, 0.0, 0.2], abs=1e-6)
