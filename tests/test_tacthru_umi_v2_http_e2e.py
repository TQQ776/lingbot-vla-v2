import http.client
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import yaml

from deploy.tacthru_umi_v2.http_server import (
    LingBotV2Backend,
    _validate_deployment_contract,
    create_http_server,
)
from deploy.tacthru_umi_v2.protocol import Observation
from deploy.tacthru_umi_v2.realman_client import (
    LingBotV2HttpClient,
    RetryableInferenceError,
    validate_server_health,
)


class FakePolicy:
    def __init__(self, *, sleep_s: float = 0.0):
        self.sleep_s = sleep_s
        self.inputs = []
        self.reset_calls = []
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()

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


def make_contract_paths(
    tmp_path: Path,
    *,
    training_norm: str = "assets/norm_stats/current.json",
    robot_norm: str = "assets/norm_stats/default.json",
) -> tuple[Path, Path, Path, Path]:
    project_root = tmp_path / "project"
    norm_stats = project_root / "assets/norm_stats/current.json"
    norm_stats.parent.mkdir(parents=True)
    norm_stats.write_text("{}\n", encoding="utf-8")
    (norm_stats.parent / "default.json").write_text("{}\n", encoding="utf-8")

    checkpoint = project_root / "output/run/checkpoints/global_step_1/hf_ckpt"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    training = {
        "data": {
            "data_name": "tacthru_umi_v2",
            "cameras": ["camera_wrist_left"],
            "norm_stats_file": training_norm,
        },
        "train": {
            "chunk_size": 50,
            "action_dim": 55,
            "max_action_dim": 55,
            "max_state_dim": 55,
        },
    }
    (checkpoint.parent.parent.parent / "lingbotvla_cli.yaml").write_text(
        yaml.safe_dump(training),
        encoding="utf-8",
    )

    robot_config_path = project_root / "configs/robot_configs/tacthru_umi_v2.yaml"
    robot_config_path.parent.mkdir(parents=True)
    robot = {
        "states": [
            {"observation.state.end.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 7}}]}},
            {"observation.state.effector.position": {"origin_keys": [{"observation.state": {"start": 7, "end": 8}}]}},
        ],
        "actions": [
            {
                "action.end.position": {
                    "origin_keys": [{"action": {"start": 0, "end": 7}}],
                    "subtract_state": True,
                    "relative_type": "quaternion_local",
                }
            },
            {
                "action.effector.position": {
                    "origin_keys": [{"action": {"start": 7, "end": 8}}],
                    "subtract_state": False,
                }
            },
        ],
        "images": ["observation.images.camera_wrist_left"],
        "norm_stats": robot_norm,
    }
    robot_config_path.write_text(yaml.safe_dump(robot), encoding="utf-8")
    return project_root, checkpoint, norm_stats, robot_config_path


def test_contract_allows_checkpoint_norm_to_override_robot_default(tmp_path: Path) -> None:
    project_root, checkpoint, norm_stats, robot_config_path = make_contract_paths(tmp_path)

    contract = _validate_deployment_contract(
        project_root=project_root,
        checkpoint=checkpoint,
        norm_stats=norm_stats,
        robot_config_path=robot_config_path,
    )

    assert contract["robot_default_norm_overridden"] is True
    assert contract["robot_default_norm_stats"].endswith("assets/norm_stats/default.json")


def test_contract_still_rejects_checkpoint_training_norm_mismatch(tmp_path: Path) -> None:
    project_root, checkpoint, norm_stats, robot_config_path = make_contract_paths(
        tmp_path,
        training_norm="assets/norm_stats/default.json",
    )

    with pytest.raises(RuntimeError, match="Norm stats path mismatch"):
        _validate_deployment_contract(
            project_root=project_root,
            checkpoint=checkpoint,
            norm_stats=norm_stats,
            robot_config_path=robot_config_path,
        )


def test_http_roundtrip_maps_exact_v2_observation_and_response_contract() -> None:
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
        for key in (
            "encode_s",
            "connect_s",
            "request_write_s",
            "response_headers_wait_s",
            "response_read_s",
            "response_parse_s",
        ):
            assert first_timing[key] >= 0.0
        server_timing = first.metadata["server_timing_s"]
        assert server_timing["request_read_s"] >= 0.0
        assert server_timing["request_decode_s"] >= 0.0
        assert server_timing["backend_lock_wait_s"] >= 0.0
        assert server_timing["backend_reset_s"] >= 0.0
        assert server_timing["backend_inference_s"] >= 0.005
        assert server_timing["backend_postprocess_s"] >= 0.0
        assert server_timing["backend_total_s"] >= 0.0
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


def test_reset_connection_avoids_reusing_health_socket_after_operator_delay() -> None:
    server = create_http_server(
        make_backend(FakePolicy()),
        host="127.0.0.1",
        port=0,
        http_keep_alive=True,
        keep_alive_idle_timeout_s=0.05,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = LingBotV2HttpClient(
        f"http://{host}:{port}", timeout_s=2.0, jpeg_quality=100, keep_alive=True
    )
    try:
        client.health()
        time.sleep(0.10)
        client.reset_connection()
        _, timing = client.predict_timed(make_observation(1), expected_steps=50)
        assert timing["connection_reused"] is False
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


def test_legacy_client_remains_compatible_with_keep_alive_server() -> None:
    server = create_http_server(
        make_backend(FakePolicy()),
        host="127.0.0.1",
        port=0,
        http_keep_alive=True,
        keep_alive_idle_timeout_s=2.0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = LingBotV2HttpClient(
        f"http://{host}:{port}", timeout_s=5.0, jpeg_quality=100, keep_alive=False
    )
    try:
        health = client.health()
        assert health["transport"]["http_keep_alive_enabled"] is True
        _, first_timing = client.predict_timed(make_observation(1), expected_steps=50)
        _, second_timing = client.predict_timed(make_observation(2), expected_steps=50)
        assert first_timing["transport"] == "legacy_urllib"
        assert second_timing["transport"] == "legacy_urllib"
        assert server.accepted_connection_count == 3
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


class _AmbiguousDisconnectConnection:
    """Connection that accepts a POST and then drops before response headers."""

    sock = object()

    def __init__(self, error: BaseException | None = None) -> None:
        self.request_calls = 0
        self.error = error or http.client.RemoteDisconnected("peer closed after request")

    def request(self, *args, **kwargs) -> None:
        self.request_calls += 1

    def getresponse(self):
        raise self.error

    def close(self) -> None:
        self.sock = None


@pytest.mark.parametrize(
    "error",
    [
        http.client.RemoteDisconnected("peer closed after request"),
        TimeoutError("timed out after request"),
    ],
)
def test_predict_is_never_replayed_after_ambiguous_disconnect(error) -> None:
    client = LingBotV2HttpClient(
        "http://127.0.0.1:18081", timeout_s=1.0, jpeg_quality=100, keep_alive=True
    )
    connection = _AmbiguousDisconnectConnection(error)
    client._connection = connection
    with pytest.raises(RuntimeError, match="Cannot reach LingBot V2 server"):
        client.predict_timed(make_observation(), expected_steps=50)
    assert connection.request_calls == 1


class _ErrorResponse:
    will_close = False
    version = 11

    def __init__(self, status: int) -> None:
        self.status = int(status)

    def read(self) -> bytes:
        return b'{"error":"temporary"}'

    def getheader(self, _name: str):
        return None


class _ErrorResponseConnection:
    sock = object()

    def __init__(self, status: int) -> None:
        self.status = int(status)
        self.request_calls = 0

    def request(self, *args, **kwargs) -> None:
        self.request_calls += 1

    def getresponse(self):
        return _ErrorResponse(self.status)

    def close(self) -> None:
        self.sock = None


@pytest.mark.parametrize("status", [408, 503])
def test_retryable_http_status_is_reported_without_replaying_post(status: int) -> None:
    client = LingBotV2HttpClient(
        "http://127.0.0.1:18081", timeout_s=1.0, jpeg_quality=100, keep_alive=True
    )
    connection = _ErrorResponseConnection(status)
    client._connection = connection
    with pytest.raises(RetryableInferenceError, match=f"HTTP {status}") as exc_info:
        client.predict_timed(make_observation(), expected_steps=50)
    assert connection.request_calls == 1
    assert exc_info.value.timing["response_status"] == status
    assert exc_info.value.timing["response_received"] is True


def test_server_http_500_remains_fatal_and_is_not_retried() -> None:
    client = LingBotV2HttpClient(
        "http://127.0.0.1:18081", timeout_s=1.0, jpeg_quality=100, keep_alive=True
    )
    connection = _ErrorResponseConnection(500)
    client._connection = connection
    with pytest.raises(RuntimeError, match="HTTP 500"):
        client.predict_timed(make_observation(), expected_steps=50)
    assert connection.request_calls == 1


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
