import json

import numpy as np
import pytest

from deploy.tacthru_umi_v2.protocol import (
    PROTOCOL_VERSION,
    ActionResponse,
    FastPredictRequest,
    Observation,
    SlowActionPlanRequest,
    SlowContextRequest,
    TactileRefineRequest,
    action_response_from_payload,
    action_response_to_payload,
    fast_predict_from_payload,
    fast_predict_to_payload,
    observation_from_payload,
    observation_to_payload,
    slow_action_plan_from_payload,
    slow_action_plan_to_payload,
    slow_context_from_payload,
    slow_context_to_payload,
    tactile_refine_from_payload,
    tactile_refine_to_payload,
)


def identity_state(gripper_width_m: float = 0.04) -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, gripper_width_m], dtype=np.float32)


def identity_actions(steps: int = 50, gripper_width_m: float = 0.03) -> np.ndarray:
    action = np.tile(identity_state(gripper_width_m), (steps, 1))
    action[:, 0] = np.linspace(0.0, 0.02, steps)
    return action


def tactile_observation(history_length: int = 8) -> Observation:
    marker = np.linspace(
        -0.2, 0.2, history_length * 48 * 2, dtype=np.float32
    ).reshape(1, history_length, 48, 2)
    tactile_rgb = np.zeros((1, 480, 640, 3), dtype=np.uint8)
    tactile_rgb[..., 1] = 180
    return Observation(
        instruction="Insert the Ethernet cable",
        state=identity_state(),
        wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        tactile_rgb=tactile_rgb,
        marker_displacement_history=marker,
        marker_valid_mask=np.ones((1, history_length, 48), dtype=np.bool_),
        marker_history_valid_mask=np.ones((1, history_length), dtype=np.bool_),
        marker_contact_state=np.asarray([2], dtype=np.int8),
        tactile_sensor_mask=np.ones((1,), dtype=np.bool_),
    )


def test_observation_roundtrip_preserves_rgb_state_ids_and_gripper_metres() -> None:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[..., 0] = 240
    observation = Observation(
        instruction="Pull the tissue",
        state=identity_state(0.04),
        wrist_rgb=image,
        request_id="request-1",
        session_id="session-1",
    )

    decoded = observation_from_payload(observation_to_payload(observation, jpeg_quality=100))

    assert decoded.request_id == "request-1"
    assert decoded.session_id == "session-1"
    assert decoded.state.shape == (8,)
    assert decoded.state[7] == pytest.approx(0.04)
    assert decoded.wrist_rgb.shape == image.shape
    assert decoded.wrist_rgb[..., 0].mean() > 220
    assert decoded.wrist_rgb[..., 2].mean() < 20


def test_protocol_rejects_version_mismatch_and_nonfinite_state() -> None:
    payload = observation_to_payload(
        Observation(instruction="Pull", state=identity_state(), wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8))
    )
    payload["protocol_version"] = PROTOCOL_VERSION + 1
    with pytest.raises(ValueError, match="protocol_version"):
        observation_from_payload(payload)

    payload["protocol_version"] = PROTOCOL_VERSION
    payload["state"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        observation_from_payload(payload)


def test_response_rejects_request_id_mismatch_and_preserves_absolute_gripper_width() -> None:
    payload = action_response_to_payload(
        action_chunk=identity_actions(gripper_width_m=0.03),
        request_id="request-1",
        session_id="session-1",
        expected_steps=50,
    )
    with pytest.raises(ValueError, match="request_id mismatch"):
        action_response_from_payload(payload, expected_request_id="other", expected_steps=50)

    response = action_response_from_payload(
        payload,
        expected_request_id="request-1",
        expected_session_id="session-1",
        expected_steps=50,
    )
    assert isinstance(response, ActionResponse)
    assert response.action_chunk.shape == (50, 8)
    assert np.allclose(response.action_chunk[:, 7], 0.03)


def test_json_serialization_refuses_nan() -> None:
    payload = {"value": float("nan")}
    with pytest.raises(ValueError):
        json.dumps(payload, allow_nan=False)


def test_wire_image_must_be_exactly_224_square() -> None:
    with pytest.raises(ValueError, match="224"):
        observation_to_payload(
            Observation(
                instruction="Pull",
                state=identity_state(),
                wrist_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            )
        )


def test_v3_tactile_roundtrip_preserves_rgb_and_marker_history_contract() -> None:
    observation = tactile_observation()

    decoded = observation_from_payload(
        observation_to_payload(observation, jpeg_quality=100)
    )

    assert decoded.tactile_rgb.shape == (1, 480, 640, 3)
    assert decoded.tactile_rgb[..., 1].mean() > 160
    assert np.allclose(
        decoded.marker_displacement_history,
        observation.marker_displacement_history,
    )
    assert decoded.marker_valid_mask.dtype == np.bool_
    assert decoded.marker_valid_mask.all()
    assert decoded.marker_history_valid_mask.all()
    assert decoded.marker_contact_state.tolist() == [2]
    assert decoded.tactile_sensor_mask.tolist() == [True]


def test_v3_tactile_roundtrip_supports_four_frame_marker_history() -> None:
    observation = tactile_observation(history_length=4)

    decoded = observation_from_payload(
        observation_to_payload(observation, jpeg_quality=100)
    )

    assert decoded.marker_displacement_history.shape == (1, 4, 48, 2)
    assert decoded.marker_valid_mask.shape == (1, 4, 48)
    assert decoded.marker_history_valid_mask.shape == (1, 4)
    assert np.allclose(
        decoded.marker_displacement_history,
        observation.marker_displacement_history,
    )


def test_tactile_protocol_rejects_invalid_contact_state() -> None:
    observation = tactile_observation()
    values = dict(observation.__dict__)
    values["marker_contact_state"] = np.asarray([99], dtype=np.int8)
    with pytest.raises(ValueError, match="contact-state"):
        observation_to_payload(Observation(**values))


def test_tactile_marker_payload_rejects_partial_contract() -> None:
    observation = tactile_observation()
    observation = Observation(
        instruction=observation.instruction,
        state=observation.state,
        wrist_rgb=observation.wrist_rgb,
        marker_displacement_history=observation.marker_displacement_history,
        tactile_sensor_mask=observation.tactile_sensor_mask,
    )
    with pytest.raises(ValueError, match="incomplete"):
        observation_to_payload(observation)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "marker_displacement_history",
            np.zeros((1, 0, 48, 2), dtype=np.float32),
            "shape",
        ),
        (
            "marker_valid_mask",
            np.ones((1, 8, 47), dtype=np.bool_),
            "shape",
        ),
        (
            "marker_history_valid_mask",
            np.ones((1, 7), dtype=np.bool_),
            "shape",
        ),
    ],
)
def test_tactile_protocol_rejects_wrong_history_shapes(field, value, match):
    observation = tactile_observation()
    values = dict(observation.__dict__)
    values[field] = value
    with pytest.raises(ValueError, match=match):
        observation_to_payload(Observation(**values))


def test_tactile_protocol_rejects_nonfinite_history_and_wrong_sample_rate():
    payload = observation_to_payload(tactile_observation())
    payload["tactile"]["marker_displacement_history"][0][0][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        observation_from_payload(payload)

    payload = observation_to_payload(tactile_observation())
    payload["tactile"]["marker_sample_hz"] = 60.0
    with pytest.raises(ValueError, match="marker_sample_hz"):
        observation_from_payload(payload)


def test_response_spec_cannot_claim_more_steps_than_payload() -> None:
    payload = action_response_to_payload(
        action_chunk=identity_actions(50),
        request_id="request-1",
        session_id="session-1",
        expected_steps=50,
    )
    payload["action_chunk"] = payload["action_chunk"][:5]
    with pytest.raises(ValueError, match="Expected 50 action steps"):
        action_response_from_payload(payload)


def test_slow_context_roundtrip_contains_both_rgb_images_and_timestamps() -> None:
    request = SlowContextRequest(
        instruction="Insert the Ethernet cable",
        wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        tactile_rgb=np.zeros((1, 480, 640, 3), dtype=np.uint8),
        tactile_sensor_mask=np.ones((1,), dtype=np.bool_),
        session_id="session-cache",
        request_id="refresh-1",
        scene_timestamp=10.0,
        tactile_rgb_timestamp=10.1,
    )

    payload = slow_context_to_payload(request, jpeg_quality=100)
    decoded = slow_context_from_payload(payload)

    assert "scene_rgb" in payload
    assert "tactile_rgb" in payload
    assert decoded.session_id == "session-cache"
    assert decoded.scene_timestamp == pytest.approx(10.0)
    assert decoded.tactile_rgb_timestamp == pytest.approx(10.1)


def test_fast_payload_is_marker_only_and_contains_no_rgb_data() -> None:
    tactile = tactile_observation(history_length=4)
    request = FastPredictRequest(
        state=tactile.state,
        marker_displacement_history=tactile.marker_displacement_history,
        marker_valid_mask=tactile.marker_valid_mask,
        marker_history_valid_mask=tactile.marker_history_valid_mask,
        marker_contact_state=tactile.marker_contact_state,
        tactile_sensor_mask=tactile.tactile_sensor_mask,
        session_id="session-cache",
        context_version=2,
        request_id="fast-1",
        marker_timestamp=10.2,
    )

    payload = fast_predict_to_payload(request)
    decoded = fast_predict_from_payload(payload)

    assert "images" not in payload
    assert "rgb" not in payload["tactile"]
    assert decoded.context_version == 2
    assert decoded.marker_displacement_history.shape == (1, 4, 48, 2)


def test_cascaded_plan_and_refine_protocol_roundtrip() -> None:
    tactile = tactile_observation(history_length=4)
    plan = SlowActionPlanRequest(
        state=tactile.state,
        session_id="session-cascade",
        context_version=3,
        action_offset=2,
        request_id="plan-1",
        timestamp=11.0,
    )
    decoded_plan = slow_action_plan_from_payload(slow_action_plan_to_payload(plan))
    assert decoded_plan.context_version == 3
    assert decoded_plan.action_offset == 2

    refine = TactileRefineRequest(
        state=tactile.state,
        marker_displacement_history=tactile.marker_displacement_history,
        marker_valid_mask=tactile.marker_valid_mask,
        marker_history_valid_mask=tactile.marker_history_valid_mask,
        marker_contact_state=tactile.marker_contact_state,
        tactile_sensor_mask=tactile.tactile_sensor_mask,
        session_id="session-cascade",
        context_version=3,
        plan_version=7,
        action_offset=4,
        request_id="refine-1",
        marker_timestamp=11.1,
    )
    payload = tactile_refine_to_payload(refine)
    decoded_refine = tactile_refine_from_payload(payload)
    assert "images" not in payload and "rgb" not in payload["tactile"]
    assert decoded_refine.plan_version == 7
    assert decoded_refine.action_offset == 4

    with pytest.raises(ValueError, match="action_offset"):
        slow_action_plan_to_payload(
            SlowActionPlanRequest(
                state=tactile.state,
                session_id="session-cascade",
                context_version=3,
                action_offset=51,
            )
        )
