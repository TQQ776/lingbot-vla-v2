import json

import numpy as np
import pytest

from deploy.tacthru_umi_v2.protocol import (
    PROTOCOL_VERSION,
    ActionResponse,
    Observation,
    action_response_from_payload,
    action_response_to_payload,
    observation_from_payload,
    observation_to_payload,
)


def identity_state(gripper_width_m: float = 0.04) -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, gripper_width_m], dtype=np.float32)


def identity_actions(steps: int = 50, gripper_width_m: float = 0.03) -> np.ndarray:
    action = np.tile(identity_state(gripper_width_m), (steps, 1))
    action[:, 0] = np.linspace(0.0, 0.02, steps)
    return action


def tactile_observation() -> Observation:
    marker = np.linspace(-0.2, 0.2, 48 * 2, dtype=np.float32).reshape(1, 48, 2)
    tactile_rgb = np.zeros((1, 480, 640, 3), dtype=np.uint8)
    tactile_rgb[..., 1] = 180
    return Observation(
        instruction="Insert the Ethernet cable",
        state=identity_state(),
        wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        tactile_rgb=tactile_rgb,
        marker_positions=marker,
        marker_reference=np.zeros_like(marker),
        previous_marker_positions=marker - 0.01,
        marker_valid_mask=np.ones((1, 48), dtype=np.bool_),
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


def test_v2_tactile_roundtrip_preserves_rgb_and_marker_contract() -> None:
    observation = tactile_observation()

    decoded = observation_from_payload(
        observation_to_payload(observation, jpeg_quality=100)
    )

    assert decoded.tactile_rgb.shape == (1, 480, 640, 3)
    assert decoded.tactile_rgb[..., 1].mean() > 160
    assert np.allclose(decoded.marker_positions, observation.marker_positions)
    assert np.allclose(
        decoded.previous_marker_positions,
        observation.previous_marker_positions,
    )
    assert decoded.marker_valid_mask.dtype == np.bool_
    assert decoded.marker_valid_mask.all()
    assert decoded.tactile_sensor_mask.tolist() == [True]


def test_tactile_marker_payload_rejects_partial_contract() -> None:
    observation = tactile_observation()
    observation = Observation(
        instruction=observation.instruction,
        state=observation.state,
        wrist_rgb=observation.wrist_rgb,
        marker_positions=observation.marker_positions,
        tactile_sensor_mask=observation.tactile_sensor_mask,
    )
    with pytest.raises(ValueError, match="incomplete"):
        observation_to_payload(observation)


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
