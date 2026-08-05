import numpy as np
import pytest


pytest.importorskip("zarr")
pytest.importorskip("scipy")

from tools.convert_tacthru_zarr_to_lerobot_v2 import (  # noqa: E402
    ACTION_CHUNK_SIZE,
    CONVERTER_VERSION,
    DATASET_FPS,
    MARKER_DISPLACEMENT_FEATURE,
    _requested_manifest,
    build_features,
    build_pose8,
    make_episode_plan,
)


def test_episode_plan_preserves_boundaries_without_loss_or_duplication() -> None:
    episode_ends = np.asarray([8, 15], dtype=np.int64)
    plan = make_episode_plan(episode_ends, num_source_episodes=2)

    assert len(plan) == 2
    assert [item["length"] for item in plan] == [8, 7]
    assert [(item["source_start"], item["source_end"]) for item in plan] == [
        (0, 8),
        (8, 15),
    ]

    covered = [
        frame_index
        for item in plan
        for frame_index in range(item["source_start"], item["source_end"])
    ]
    assert covered == list(range(15))


def test_episode_plan_keeps_native_contiguous_frames() -> None:
    plan = make_episode_plan(np.asarray([60], dtype=np.int64), num_source_episodes=1)
    episode = plan[0]
    first_fifty = list(range(episode["source_start"], episode["source_end"]))[:50]
    assert first_fifty == list(range(50))


def test_manifest_declares_v2_native_temporal_layout() -> None:
    plan = make_episode_plan(np.asarray([60, 115], dtype=np.int64), num_source_episodes=2)
    manifest = _requested_manifest(
        signature={"source": "/tmp/source.zarr"},
        task="Pull the tissue",
        repo_id="test-dataset",
        source_episodes=2,
        source_frames=115,
        plan=plan,
        wrist_shape=(224, 224, 3),
        excluded_tactile_source_keys=["tacthru_l_rgb", "tacthru_l_marker"],
    )

    assert manifest["converter"] == "tacthru_umi_v2_native_30hz"
    assert manifest["converter_version"] == CONVERTER_VERSION == 6
    assert manifest["source_fps"] == manifest["output_fps"] == DATASET_FPS == 30
    assert manifest["source_episodes"] == manifest["output_episodes"] == 2
    assert manifest["source_frames"] == manifest["output_frames"] == 115
    assert manifest["output_episode_lengths"] == [60, 55]
    assert manifest["action_chunk_size"] == ACTION_CHUNK_SIZE == 50
    assert manifest["valid_anchors"] == 11 + 6
    assert manifest["episode_mapping"] == "one_source_episode_to_one_output_episode"
    assert manifest["frame_sampling"] == "contiguous_stride_1"
    assert manifest["frames_dropped"] == manifest["frames_duplicated"] == 0
    assert manifest["visual_input_policy"] == "wrist_rgb_only"
    assert manifest["image_features"] == ["observation.images.camera_wrist_left"]
    assert manifest["wrist_rgb_source_key"] == "camera0_rgb"
    assert manifest["tactile_inputs"]["source_keys_present"] == [
        "tacthru_l_marker",
        "tacthru_l_rgb",
    ]
    assert manifest["tactile_inputs"]["enabled"] is False
    assert manifest["tactile_inputs"]["copied_to_lerobot"] is False
    assert manifest["tactile_inputs"]["used_for_training"] is False
    assert "phase_count" not in manifest
    assert "phase_stride" not in manifest


def test_manifest_declares_vtla_tactile_contract_when_enabled() -> None:
    plan = make_episode_plan(np.asarray([60], dtype=np.int64), num_source_episodes=1)
    manifest = _requested_manifest(
        signature={"source": "/tmp/source.zarr"},
        task="Insert the Ethernet cable",
        repo_id="vtla-dataset",
        source_episodes=1,
        source_frames=60,
        plan=plan,
        wrist_shape=(224, 224, 3),
        excluded_tactile_source_keys=["tacthru_l_rgb", "tacthru_l_marker"],
        include_tactile=True,
        tactile_info={"tactile_rgb_shape": (224, 224, 3), "num_markers": 48},
    )
    tactile = manifest["tactile_inputs"]
    assert tactile["enabled"] is True
    assert tactile["copied_to_lerobot"] is True
    assert tactile["num_markers"] == 48
    assert tactile["marker_representation"] == "normalized_displacement"
    assert tactile["formula"] == (
        "2 * (current_xy - reference_xy) / [image_width, image_height]"
    )
    assert tactile["history_storage"] == "per_frame_displacement"
    assert tactile["history_construction"] == "dataset_same_episode_offsets"
    assert tactile["history_length"] == 8
    assert tactile["history_padding"] == "earliest_valid_frame_replication"
    assert tactile["marker_sample_hz"] == 30
    assert "observation.images.tactile_left" in manifest["image_features"]


def test_marker_displacement_feature_has_exact_rank_two_schema() -> None:
    features = build_features(
        (224, 224, 3),
        {"tactile_rgb_shape": (480, 640, 3), "num_markers": 48},
    )
    marker = features[MARKER_DISPLACEMENT_FEATURE]
    assert marker["shape"] == (48, 2)
    assert marker["names"] is None


def test_pose8_is_xyzw_normalized_and_canonical() -> None:
    position = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    rotvec = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0 * np.pi]], dtype=np.float64)
    gripper = np.asarray([[0.01], [0.02]], dtype=np.float32)

    pose = build_pose8(position, rotvec, gripper)

    assert pose.shape == (2, 8)
    np.testing.assert_allclose(pose[0, 3:7], [0.0, 0.0, 0.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(pose[:, 3:7], axis=-1), 1.0, atol=1e-7)
    assert np.all(pose[:, 6] >= 0.0)
    np.testing.assert_allclose(pose[:, :3], position)
    np.testing.assert_allclose(pose[:, 7:], gripper)
