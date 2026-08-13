import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.tacthru_umi_v2.http_server import (
    LingBotV2Backend,
    _resolve_training_project_path,
    create_http_server,
)
from deploy.tacthru_umi_v2.protocol import Observation
from deploy.tacthru_umi_v2.realman_client import LingBotV2HttpClient, validate_server_health


class FakePolicy:
    def __init__(self, *, sleep_s: float = 0.0, tactile: dict | None = None):
        self.sleep_s = sleep_s
        self.inputs = []
        self.reset_calls = []
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()
        if tactile is not None:
            self.config = SimpleNamespace(tactile=tactile)

    def reset(self, robo_name):
        self.reset_calls.append(robo_name)

    def infer(self, observation):
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.inputs.append(observation)
            time.sleep(self.sleep_s)
            actions = np.tile(
                np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.03], dtype=np.float32),
                (50, 1),
            )
            return {"action": actions}
        finally:
            with self.guard:
                self.active -= 1


def make_backend(policy: FakePolicy) -> LingBotV2Backend:
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


def test_training_artifact_path_rebases_only_matching_project_root(tmp_path) -> None:
    project_root = tmp_path / "lingbot-vla-v2"
    expected = project_root / "assets/norm_stats/example.json"
    saved = Path(
        "/root/kube-user/ns/example/lingbot-vla-v2/assets/norm_stats/example.json"
    )
    assert _resolve_training_project_path(project_root, saved) == expected.resolve()

    external = Path("/opt/checkpoints/example.json")
    assert _resolve_training_project_path(project_root, external) == external


def make_observation(index: int = 0) -> Observation:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[..., 1] = 50 + index
    return Observation(
        instruction="Pull the tissue",
        state=np.asarray([0, 0, 0, 0, 0, 0, 1, 0.04], dtype=np.float32),
        wrist_rgb=image,
        request_id=f"request-{index}",
        session_id="session-1",
        metadata={"episode_reset": index == 0},
    )


def test_compile_warmup_uses_the_real_task_instruction() -> None:
    policy = FakePolicy()
    backend = make_backend(policy)

    result = backend.warmup(instruction="Insert the Ethernet cable.")

    assert result["instruction"] == "Insert the Ethernet cable."
    assert result["action_shape"] == [50, 8]
    assert policy.inputs[-1]["task"] == "Insert the Ethernet cable."


def tactile_contract(*, gated: bool = False, history_length: int = 8) -> dict:
    contract = {
        "enabled": True,
        "num_sensors": 1,
        "num_markers": 48,
        "use_rgb": True,
        "use_markers": True,
        "marker_history_length": history_length,
        "marker_sample_hz": 30.0,
        "marker_feature_mode": "displacement_history",
        "rgb_keys": ["observation.images.tactile_left"],
        "marker_displacement_keys": [
            "observation.tactile.marker_displacement_left"
        ],
        "marker_valid_mask_keys": ["observation.tactile.marker_valid_left"],
    }
    if gated:
        contract.update(
            marker_contact_gate={
                "enabled": True,
                "mode": "hard",
                "target": "marker_only",
                "point_threshold": 0.1,
                "on_threshold": 0.2,
                "off_threshold": 0.1,
            },
            gate_tactile_rgb=False,
        )
    return contract


def make_tactile_observation(history_length: int = 8) -> Observation:
    marker = np.full((1, history_length, 48, 2), 0.2, dtype=np.float32)
    return Observation(
        instruction="Insert the Ethernet cable",
        state=np.asarray([0, 0, 0, 0, 0, 0, 1, 0.04], dtype=np.float32),
        wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        tactile_rgb=np.full((1, 480, 640, 3), 30, dtype=np.uint8),
        marker_displacement_history=marker,
        marker_valid_mask=np.ones((1, history_length, 48), dtype=np.bool_),
        marker_history_valid_mask=np.ones((1, history_length), dtype=np.bool_),
        tactile_sensor_mask=np.ones((1,), dtype=np.bool_),
        request_id="request-tactile",
        session_id="session-tactile",
        metadata={"episode_reset": True},
    )


def test_http_roundtrip_maps_exact_v3_observation_and_response_contract() -> None:
    policy = FakePolicy()
    server = create_http_server(make_backend(policy), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = LingBotV2HttpClient(f"http://{host}:{port}", timeout_s=5.0, jpeg_quality=100)
        chunk_size = validate_server_health(client.health())
        observation = make_observation()
        response = client.predict(observation, expected_steps=chunk_size)

        assert response.request_id == observation.request_id
        assert response.action_chunk.shape == (50, 8)
        assert np.allclose(response.action_chunk[:, 0], 0.01)
        assert np.allclose(response.action_chunk[:, 7], 0.03)
        assert set(policy.inputs[0]) == {
            "observation.state",
            "observation.images.camera_wrist_left",
            "task",
        }
        assert policy.inputs[0]["observation.state"].shape == (8,)
        assert policy.inputs[0]["observation.state"][7] == pytest.approx(0.04)
        assert policy.inputs[0]["observation.images.camera_wrist_left"].shape == (224, 224, 3)
        assert policy.inputs[0]["task"] == "Pull the tissue"
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_reset_connection_discards_idle_health_socket_before_prediction() -> None:
    policy = FakePolicy()
    server = create_http_server(
        make_backend(policy),
        host="127.0.0.1",
        port=0,
        http_keep_alive=True,
        keep_alive_idle_timeout_s=0.05,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = LingBotV2HttpClient(
            f"http://{host}:{port}",
            timeout_s=5.0,
            jpeg_quality=100,
            keep_alive=True,
        )
        validate_server_health(client.health())
        time.sleep(0.1)

        client.reset_connection()
        response = client.predict(make_observation(), expected_steps=50)

        assert response.action_chunk.shape == (50, 8)
        assert policy.inputs[-1]["task"] == "Pull the tissue"
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_http_roundtrip_forwards_vtla_inputs_instead_of_dropping_them() -> None:
    policy = FakePolicy(tactile=tactile_contract())
    server = create_http_server(make_backend(policy), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = LingBotV2HttpClient(
            f"http://{host}:{port}", timeout_s=5.0, jpeg_quality=100
        )
        health = client.health()
        assert health["tactile_enabled"] is True
        expected_contract = {
            "enabled": True,
            "num_sensors": 1,
            "num_markers": 48,
            "use_rgb": True,
            "use_markers": True,
            "marker_history_length": 8,
            "marker_sample_hz": 30.0,
        }
        assert expected_contract.items() <= health["tactile"].items()
        assert health["tactile"]["marker_tokens_per_sensor"] == 8
        assert health["tactile"]["marker_contact_gate"]["mode"] == "none"
        chunk_size = validate_server_health(health)
        response = client.predict(
            make_tactile_observation(), expected_steps=chunk_size
        )

        assert response.request_id == "request-tactile"
        model_input = policy.inputs[0]
        assert model_input["tactile_rgb"].shape == (1, 480, 640, 3)
        assert model_input["marker_displacement_history"].shape == (
            1, 8, 48, 2
        )
        assert np.allclose(model_input["marker_displacement_history"], 0.2)
        assert model_input["marker_valid_mask"].all()
        assert model_input["marker_history_valid_mask"].all()
        assert model_input["tactile_sensor_mask"].tolist() == [True]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_backend_accepts_four_frame_history_and_rejects_wrong_length() -> None:
    backend = make_backend(
        FakePolicy(tactile=tactile_contract(history_length=4))
    )

    backend.predict(make_tactile_observation(history_length=4))
    assert backend.policy.inputs[-1]["marker_displacement_history"].shape == (
        1,
        4,
        48,
        2,
    )
    with pytest.raises(ValueError, match="does not match checkpoint"):
        backend.predict(make_tactile_observation(history_length=8))


def test_vtla_backend_rejects_request_that_omits_checkpoint_modalities() -> None:
    backend = make_backend(FakePolicy(tactile=tactile_contract()))

    with pytest.raises(ValueError, match="does not match checkpoint"):
        backend.predict(make_observation())


def test_gated_backend_requires_and_forwards_contact_state() -> None:
    policy = FakePolicy(tactile=tactile_contract(gated=True))
    backend = make_backend(policy)
    observation = make_tactile_observation()
    with pytest.raises(ValueError, match="does not match checkpoint"):
        backend.predict(observation)

    values = dict(observation.__dict__)
    values["marker_contact_state"] = np.asarray([1], dtype=np.int8)
    backend.predict(Observation(**values))

    assert policy.inputs[-1]["marker_contact_state"].tolist() == [1]


def test_vision_backend_rejects_unexpected_tactile_request() -> None:
    backend = make_backend(FakePolicy())

    with pytest.raises(ValueError, match="tactile-disabled"):
        backend.predict(make_tactile_observation())


def test_backend_serializes_concurrent_policy_inference() -> None:
    policy = FakePolicy(sleep_s=0.03)
    backend = make_backend(policy)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(backend.predict, [make_observation(index) for index in range(4)]))

    assert policy.max_active == 1
    assert [result["request_id"] for result in results] == [f"request-{index}" for index in range(4)]


def test_keep_alive_reuses_one_connection_and_reports_stage_timings() -> None:
    policy = FakePolicy(sleep_s=0.01)
    server = create_http_server(
        make_backend(policy),
        host="127.0.0.1",
        port=0,
        http_keep_alive=True,
        keep_alive_idle_timeout_s=2.0,
        keep_alive_max_requests=20,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = LingBotV2HttpClient(
        f"http://{host}:{port}",
        timeout_s=5.0,
        jpeg_quality=100,
        keep_alive=True,
    )
    try:
        health = client.health()
        assert health["transport"]["http_protocol"] == "HTTP/1.1"
        assert health["transport"]["http_keep_alive_enabled"] is True

        first, first_timing = client.predict_timed(make_observation(1), expected_steps=50)
        second, second_timing = client.predict_timed(make_observation(2), expected_steps=50)

        assert first.request_id == "request-1"
        assert second.request_id == "request-2"
        assert server.accepted_connection_count == 1
        assert first_timing["transport"] == "http_keep_alive"
        assert first_timing["connection_reused"] is True
        assert second_timing["connection_reused"] is True
        assert first_timing["server_will_close"] is False
        assert first_timing["server_timing_s"]["backend_policy_infer_s"] >= 0.005
        assert first_timing["server_timing_s"]["response_encode_s"] >= 0.0
        server_timing = first.metadata["server_timing_s"]
        assert server_timing["request_read_s"] >= 0.0
        assert server_timing["request_decode_s"] >= 0.0
        assert server_timing["backend_lock_wait_s"] >= 0.0
        assert server_timing["backend_inference_s"] >= 0.005
        assert first.metadata["request_trace_id"] == first_timing["request_trace_id"]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_close_mode_and_old_server_compatibility_are_explicitly_available() -> None:
    policy = FakePolicy()
    server = create_http_server(make_backend(policy), host="127.0.0.1", port=0, http_keep_alive=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = LingBotV2HttpClient(
        f"http://{host}:{port}",
        timeout_s=5.0,
        jpeg_quality=100,
        keep_alive=False,
    )
    try:
        client.health()
        client.predict(make_observation(1), expected_steps=50)
        client.predict(make_observation(2), expected_steps=50)
        assert server.accepted_connection_count == 3
        assert client.last_request_timing["transport"] == "legacy_urllib"
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_keep_alive_server_closes_at_request_limit_and_client_reconnects_next_request() -> None:
    policy = FakePolicy()
    server = create_http_server(
        make_backend(policy),
        host="127.0.0.1",
        port=0,
        http_keep_alive=True,
        keep_alive_idle_timeout_s=2.0,
        keep_alive_max_requests=2,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = LingBotV2HttpClient(
        f"http://{host}:{port}",
        timeout_s=5.0,
        jpeg_quality=100,
        keep_alive=True,
    )
    try:
        client.health()
        _, first_timing = client.predict_timed(make_observation(1), expected_steps=50)
        _, second_timing = client.predict_timed(make_observation(2), expected_steps=50)
        assert first_timing["server_will_close"] is True
        assert second_timing["connection_reused"] is False
        assert server.accepted_connection_count == 2
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_keep_alive_client_remains_compatible_with_close_only_server() -> None:
    server = create_http_server(
        make_backend(FakePolicy()),
        host="127.0.0.1",
        port=0,
        http_keep_alive=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = LingBotV2HttpClient(
        f"http://{host}:{port}", timeout_s=5.0, jpeg_quality=100, keep_alive=True
    )
    try:
        client.health()
        _, first_timing = client.predict_timed(make_observation(1), expected_steps=50)
        _, second_timing = client.predict_timed(make_observation(2), expected_steps=50)
        assert first_timing["server_will_close"] is True
        assert second_timing["connection_reused"] is False
        assert server.accepted_connection_count == 3
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_idle_keep_alive_connection_does_not_hold_request_slot() -> None:
    server = create_http_server(
        make_backend(FakePolicy()),
        host="127.0.0.1",
        port=0,
        max_request_threads=1,
        http_keep_alive=True,
        keep_alive_idle_timeout_s=2.0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    first = LingBotV2HttpClient(
        f"http://{host}:{port}", timeout_s=5.0, jpeg_quality=100, keep_alive=True
    )
    second = LingBotV2HttpClient(
        f"http://{host}:{port}", timeout_s=5.0, jpeg_quality=100, keep_alive=True
    )
    try:
        assert first.health()["ready"] is True
        assert second.health()["ready"] is True
        assert server.accepted_connection_count == 2
    finally:
        first.close()
        second.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_http_client_close_is_idempotent_and_prevents_reuse() -> None:
    client = LingBotV2HttpClient(
        "http://127.0.0.1:9",
        timeout_s=0.1,
        jpeg_quality=90,
        keep_alive=True,
    )
    client.close()
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.health()
