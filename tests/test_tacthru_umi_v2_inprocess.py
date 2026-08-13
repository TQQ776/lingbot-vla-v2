from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.tacthru_umi_v2.http_server import LingBotV2Backend
from deploy.tacthru_umi_v2.protocol import CAMERA_KEY, Observation
from deploy.tacthru_umi_v2.realman_client import (
    LingBotV2InProcessClient,
    _build_inference_client,
    build_parser,
)


class FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(tactile=None)
        self.inputs = []
        self.reset_calls = []

    def reset(self, robo_name: str) -> None:
        self.reset_calls.append(robo_name)

    def infer(self, observation: dict) -> dict:
        self.inputs.append(observation)
        action = np.tile(
            np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.03], dtype=np.float32),
            (50, 1),
        )
        return {"action": action}


def _backend(policy: FakePolicy) -> LingBotV2Backend:
    return LingBotV2Backend(
        policy,
        checkpoint=Path("/tmp/fake/hf_ckpt"),
        norm_stats=Path("/tmp/fake/norm.json"),
        robot_config_path=Path("/tmp/fake/tacthru_umi_v2.yaml"),
        chunk_size=50,
        use_compile=False,
        dtype="bf16",
        inference_lock_timeout_s=2.0,
    )


def _observation() -> Observation:
    return Observation(
        instruction="Insert the Ethernet cable",
        state=np.asarray([0, 0, 0, 0, 0, 0, 1, 0.004], dtype=np.float32),
        wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        request_id="request-1",
        session_id="session-1",
        metadata={"episode_reset": True},
    )


def test_inprocess_client_bypasses_wire_serialization_and_preserves_contract() -> None:
    policy = FakePolicy()
    client = LingBotV2InProcessClient(_backend(policy))
    observation = _observation()

    response, timing = client.predict_timed(observation, expected_steps=50)

    assert np.shares_memory(policy.inputs[0][CAMERA_KEY], observation.wrist_rgb)
    assert response.action_chunk.shape == (50, 8)
    assert response.request_id == observation.request_id
    assert response.session_id == observation.session_id
    assert response.metadata["backend"] == "lingbot-vla-v2-tacthru-umi-inprocess"
    assert response.metadata["transport"] == "inprocess_direct"
    assert timing["transport"] == "inprocess"
    assert timing["encode_s"] == 0.0
    assert timing["response_parse_s"] == 0.0
    assert client.health()["transport"]["serialization"] == "none"


def test_backend_http_payload_entry_point_remains_available() -> None:
    backend = _backend(FakePolicy())

    payload = backend.predict(_observation())

    assert payload["protocol"] == "lingbot-vla-v2-tacthru-umi"
    assert payload["request_id"] == "request-1"
    assert len(payload["action_chunk"]) == 50


def test_transport_defaults_to_http_and_inprocess_does_not_require_url() -> None:
    http_args = build_parser().parse_args(
        ["synthetic", "--server-url", "http://127.0.0.1:18081"]
    )
    direct_args = build_parser().parse_args(
        ["synthetic", "--transport", "inprocess", "--checkpoint", "/tmp/hf_ckpt"]
    )

    assert http_args.transport == "http"
    assert http_args.server_url == "http://127.0.0.1:18081"
    assert direct_args.transport == "inprocess"
    assert direct_args.server_url is None
    assert direct_args.warmup is True


def test_http_transport_still_requires_server_url() -> None:
    args = build_parser().parse_args(["health"])

    with pytest.raises(ValueError, match="--server-url is required"):
        _build_inference_client(args)
