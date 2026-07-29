import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def discover_serial_ports() -> list[str]:
    candidates = []
    seen = set()
    for pattern in ("/dev/serial/by-id/*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            if path.exists():
                path_str = str(path)
                real_path = os.path.realpath(path_str)
                key = (path_str, real_path)
                if key not in seen:
                    seen.add(key)
                    candidates.append(path_str)
    return candidates


def describe_port(port: str) -> str:
    real_path = os.path.realpath(port)
    access = []
    access.append("r" if os.access(port, os.R_OK) else "-")
    access.append("w" if os.access(port, os.W_OK) else "-")
    return f"{port} -> {real_path} access={''.join(access)}"


def port_priority(port: str) -> tuple[int, str]:
    name = port.lower()
    score = 0

    if "hdsc" in name or "cdc_device" in name:
        score += 100
    if "/dev/ttyacm" in name or "/dev/ttyusb" in name:
        score += 40
    if "/dev/serial/by-id/" in name:
        score += 20
    if "espressif" in name or "jtag" in name or "debug" in name:
        score -= 200

    return (-score, name)


def get_candidate_ports(requested_port: str) -> list[str]:
    if requested_port.lower() != "auto":
        return [requested_port]

    candidates = discover_serial_ports()
    if not candidates:
        raise FileNotFoundError(
            "No serial ports were detected under /dev/serial/by-id, /dev/ttyACM*, or /dev/ttyUSB*."
        )

    return sorted(candidates, key=port_priority)


def format_info(info: dict) -> str:
    if not info.get("valid", False):
        return "valid=False"

    return (
        f"valid=True "
        f"width={float(info['width']):.5f}m "
        f"pos={float(info['pos']):.5f}m "
        f"target={float(info['target']):.5f}m "
        f"current={float(info['current']):.3f}A "
        f"temp={int(info['temperature'])} "
        f"error={int(info['error'])}"
    )


def send_target_and_sample(
    cyl: Any,
    target: float,
    control_cylinder: bool,
    duration_sec: float,
    sample_period_sec: float,
    label: str,
) -> tuple[int, int]:
    deadline = time.monotonic() + duration_sec
    num_sent = 0
    num_valid = 0

    while True:
        info = cyl.set_target_with_feedback(target, control_cylinder=control_cylinder)
        num_sent += 1
        if info.get("valid", False):
            num_valid += 1
        print(f"[{label}] sample={num_sent:02d} {format_info(info)}", flush=True)

        if time.monotonic() >= deadline:
            break
        time.sleep(sample_period_sec)

    return num_sent, num_valid


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test serial open/close control for the gripper cylinder.")
    parser.add_argument("--port", default="auto", help="Serial port path, or 'auto' to pick the first detected port.")
    parser.add_argument("--baudrate", type=int, default=921600, help="Serial baudrate.")
    parser.add_argument("--device-id", type=int, default=1, help="Cylinder device id.")
    parser.add_argument("--max-q", type=float, default=0.03, help="Cylinder maximum stroke used by CylinderCOMM.")
    parser.add_argument(
        "--width-mapping",
        type=Path,
        default=REPO_ROOT / "data/gripper/width_mapping.json",
        help="Path to gripper width mapping json.",
    )
    parser.add_argument("--open-width", type=float, default=0.072, help="Open target width in meters.")
    parser.add_argument("--close-width", type=float, default=0.0, help="Close target width in meters.")
    parser.add_argument("--cycles", type=int, default=3, help="Number of close/open cycles to run.")
    parser.add_argument(
        "--move-duration",
        type=float,
        default=1.0,
        help="How long to keep sending the same target for each open/close phase.",
    )
    parser.add_argument(
        "--sample-period",
        type=float,
        default=0.1,
        help="Delay between repeated feedback commands while holding a target.",
    )
    parser.add_argument(
        "--start-open",
        action="store_true",
        help="Send an initial open command before the first cycle.",
    )
    parser.add_argument(
        "--control-cylinder",
        action="store_true",
        help="Interpret open/close targets as raw cylinder displacement instead of gripper width.",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Only list detected serial ports and exit.",
    )
    return parser


def main():
    args = build_argparser().parse_args()

    candidates = discover_serial_ports()
    if candidates:
        print("Detected serial ports:", flush=True)
        for port in sorted(candidates, key=port_priority):
            print(f"  - {describe_port(port)}", flush=True)
    else:
        print("Detected serial ports: []", flush=True)

    if args.list_ports:
        return

    if not args.width_mapping.exists():
        raise FileNotFoundError(f"Width mapping file not found: {args.width_mapping}")

    try:
        from real_world.grippers.comm import CylinderCOMM
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import CylinderCOMM dependencies. Please run this script with the project "
            "environment that includes pyserial, for example `.venv/bin/python`."
        ) from exc

    candidate_ports = get_candidate_ports(args.port)
    last_error = None

    for idx, port in enumerate(candidate_ports, start=1):
        print(
            f"Trying port {idx}/{len(candidate_ports)}: {describe_port(port)} "
            f"baudrate={args.baudrate} device_id={args.device_id}",
            flush=True,
        )

        total_sent = 0
        total_valid = 0

        try:
            with CylinderCOMM(
                serial_path=port,
                max_q=args.max_q,
                id=args.device_id,
                baudrate=args.baudrate,
                width_mapping_path=str(args.width_mapping),
            ) as cyl:
                init_info = cyl.initialize_with_feedback()
                print(f"[init] {format_info(init_info)}", flush=True)

                if not init_info.get("valid", False):
                    raise RuntimeError(
                        f"Port {port} opened successfully but did not return valid gripper feedback."
                    )

                if args.start_open:
                    sent, valid = send_target_and_sample(
                        cyl=cyl,
                        target=args.open_width,
                        control_cylinder=args.control_cylinder,
                        duration_sec=args.move_duration,
                        sample_period_sec=args.sample_period,
                        label="warmup-open",
                    )
                    total_sent += sent
                    total_valid += valid

                for cycle_idx in range(args.cycles):
                    print(f"=== cycle {cycle_idx + 1}/{args.cycles}: close -> open ===", flush=True)

                    sent, valid = send_target_and_sample(
                        cyl=cyl,
                        target=args.close_width,
                        control_cylinder=args.control_cylinder,
                        duration_sec=args.move_duration,
                        sample_period_sec=args.sample_period,
                        label=f"cycle{cycle_idx + 1}-close",
                    )
                    total_sent += sent
                    total_valid += valid

                    sent, valid = send_target_and_sample(
                        cyl=cyl,
                        target=args.open_width,
                        control_cylinder=args.control_cylinder,
                        duration_sec=args.move_duration,
                        sample_period_sec=args.sample_period,
                        label=f"cycle{cycle_idx + 1}-open",
                    )
                    total_sent += sent
                    total_valid += valid

            print(
                f"Done on port {port}. valid_feedback={total_valid}/{total_sent} "
                f"({(100.0 * total_valid / max(total_sent, 1)):.1f}%).",
                flush=True,
            )
            return
        except Exception as exc:
            last_error = exc
            print(f"Port {port} failed: {type(exc).__name__}: {exc}", flush=True)

            if args.port.lower() != "auto":
                raise

    raise RuntimeError(
        "Failed to find a working gripper serial port automatically. "
        "Please check power, permissions, baudrate, device id, or pass --port explicitly."
    ) from last_error


if __name__ == "__main__":
    main()
