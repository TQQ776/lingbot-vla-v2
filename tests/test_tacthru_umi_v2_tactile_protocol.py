import time

import numpy as np
import pytest

from deploy.tacthru_umi_v2.protocol import (
    PROTOCOL_VERSION_V1,
    PROTOCOL_VERSION_V2,
    Observation,
    action_response_from_payload,
    action_response_to_payload,
    observation_from_payload,
    observation_to_payload,
)


def _state() -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.004], dtype=np.float32)


def _v2_observation(*, history_steps: int = 4) -> Observation:
    now = time.time()
    rgb = np.zeros((history_steps, 224, 224, 3), dtype=np.uint8)
    rgb[:, :, :, 1] = np.arange(history_steps, dtype=np.uint8)[:, None, None]
    marker = np.zeros((history_steps, 48, 2), dtype=np.float32)
    marker[:, :, 0] = np.linspace(0.0, 0.02, history_steps)[:, None]
    history_mask = np.asarray([False] * (history_steps - 2) + [True, True], dtype=np.bool_)
    marker_valid = np.ones((history_steps, 48), dtype=np.bool_)
    marker_valid[~history_mask] = False
    return Observation(
        instruction="Insert the Ethernet cable.",
        state=_state(),
        wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        request_id="request-v2",
        session_id="session-v2",
        timestamp=now,
        protocol_version=PROTOCOL_VERSION_V2,
        contract_sha256="a" * 64,
        wrist_timestamp=now,
        tactile_rgb_history=rgb,
        tactile_rgb_timestamps=np.full((history_steps,), now, dtype=np.float64),
        tactile_rgb_history_mask=history_mask,
        marker_flow=marker,
        marker_valid_mask=marker_valid,
        marker_timestamps=np.full((history_steps,), now, dtype=np.float64),
        marker_history_mask=history_mask,
        metadata={"dry_run": True},
    )


def test_protocol_v2_roundtrip_preserves_independent_tactile_modalities_and_masks() -> None:
    observation = _v2_observation()

    payload = observation_to_payload(observation, jpeg_quality=100)
    decoded = observation_from_payload(payload)

    assert payload["protocol_version"] == PROTOCOL_VERSION_V2
    assert payload["schema_version"] == PROTOCOL_VERSION_V2
    assert payload["modality_presence"] == {
        "wrist_rgb": True,
        "tactile_rgb": True,
        "tactile_marker": True,
    }
    assert decoded.protocol_version == PROTOCOL_VERSION_V2
    assert decoded.contract_sha256 == "a" * 64
    assert decoded.tactile_rgb_history.shape == (4, 224, 224, 3)
    assert decoded.tactile_rgb_history_mask.tolist() == [False, False, True, True]
    assert decoded.marker_flow.shape == (4, 48, 2)
    assert decoded.marker_valid_mask[:2].sum() == 0
    assert decoded.marker_valid_mask[2:].all()


def test_protocol_v1_payload_is_unchanged_and_rejects_v2_fields() -> None:
    observation = Observation(
        instruction="Insert",
        state=_state(),
        wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        protocol_version=PROTOCOL_VERSION_V1,
    )
    payload = observation_to_payload(observation)
    assert "schema_version" not in payload
    assert "tactile_rgb_history" not in payload
    payload["marker_flow"] = []
    with pytest.raises(ValueError, match="Protocol v1 request contains v2 tactile fields"):
        observation_from_payload(payload)


def test_protocol_v2_rejects_inconsistent_presence_and_invalid_marker_padding() -> None:
    payload = observation_to_payload(_v2_observation())
    payload["modality_presence"]["tactile_marker"] = False
    with pytest.raises(ValueError, match="modality_presence"):
        observation_from_payload(payload)

    observation = _v2_observation()
    invalid = np.asarray(observation.marker_valid_mask).copy()
    invalid[0, 0] = True
    with pytest.raises(ValueError, match="padded/invalid"):
        observation_to_payload(
            Observation(**{**observation.__dict__, "marker_valid_mask": invalid})
        )


def test_protocol_v2_action_response_uses_matching_version() -> None:
    actions = np.tile(_state(), (50, 1))
    payload = action_response_to_payload(
        action_chunk=actions,
        request_id="request-v2",
        session_id="session-v2",
        expected_steps=50,
        protocol_version=PROTOCOL_VERSION_V2,
    )
    response = action_response_from_payload(payload, expected_steps=50)
    assert payload["schema_version"] == PROTOCOL_VERSION_V2
    assert response.protocol_version == PROTOCOL_VERSION_V2
