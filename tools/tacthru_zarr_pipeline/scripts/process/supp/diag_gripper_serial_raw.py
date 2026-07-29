import argparse
import os
import sys
import time
from pathlib import Path


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


def to_hex(data: bytes) -> str:
    return " ".join(f"{byte:02X}" for byte in data)


def parse_hex_string(raw: str) -> bytes:
    cleaned = raw.replace(",", " ").replace("0x", "").replace("0X", "").strip()
    if not cleaned:
        return b""
    parts = [part for part in cleaned.split() if part]
    return bytes(int(part, 16) for part in parts)


def compute_checksum(packet_wo_checksum: bytes) -> int:
    return sum(packet_wo_checksum[2:]) & 0xFF


def build_packet(device_id: int, cmd: int, addr: int | None = None, values: list[int] | None = None) -> bytes:
    values = [] if values is None else list(values)
    length = len(values) + 2
    payload = [0x55, 0xAA, length, device_id & 0xFF, cmd & 0xFF]
    if addr is not None:
        payload.append(addr & 0xFF)
    payload.extend(v & 0xFF for v in values)
    payload.append(compute_checksum(bytes(payload)))
    return bytes(payload)


def build_preset_packet(name: str, device_id: int) -> bytes:
    name = name.lower()
    if name == "clear-error":
        return build_packet(device_id=device_id, cmd=0x04, addr=0x00, values=[0x1E])
    if name == "follow-open":
        val = 2000
        return build_packet(device_id=device_id, cmd=0x20, addr=0x37, values=[val & 0xFF, (val >> 8) & 0xFF])
    if name == "follow-close":
        val = 0
        return build_packet(device_id=device_id, cmd=0x20, addr=0x37, values=[val & 0xFF, (val >> 8) & 0xFF])
    raise ValueError(f"Unknown preset: {name}")


def read_until_quiet(ser, timeout: float, quiet_time: float) -> bytes:
    deadline = time.monotonic() + timeout
    recv = bytearray()
    last_byte_time = None

    while time.monotonic() < deadline:
        chunk = ser.read_all()
        if chunk:
            recv.extend(chunk)
            last_byte_time = time.monotonic()
        elif last_byte_time is not None and (time.monotonic() - last_byte_time) >= quiet_time:
            break
        time.sleep(0.001)

    return bytes(recv)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raw hex serial diagnostic tool for the gripper cylinder.")
    parser.add_argument("--port", required=True, help="Serial port path.")
    parser.add_argument("--baudrate", type=int, default=921600, help="Serial baudrate.")
    parser.add_argument("--timeout", type=float, default=0.2, help="Max seconds to wait for response bytes.")
    parser.add_argument("--quiet-time", type=float, default=0.02, help="Stop reading after this much idle time once bytes have started arriving.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to send the packet.")
    parser.add_argument("--gap", type=float, default=0.2, help="Seconds to wait between repeated sends.")
    parser.add_argument("--device-id", type=int, default=1, help="Device id used by preset packets.")
    parser.add_argument(
        "--preset",
        choices=["clear-error", "follow-open", "follow-close"],
        default=None,
        help="Send a known gripper packet instead of manual hex bytes.",
    )
    parser.add_argument(
        "--hex",
        default="",
        help="Manual hex bytes to send, e.g. '55 AA 03 01 04 00 1E 26'. Ignored when --preset is provided.",
    )
    parser.add_argument("--list-ports", action="store_true", help="List detected serial ports and exit.")
    return parser


def main():
    args = build_argparser().parse_args()

    ports = discover_serial_ports()
    if ports:
        print("Detected serial ports:", flush=True)
        for port in ports:
            print(f"  - {describe_port(port)}", flush=True)
    else:
        print("Detected serial ports: []", flush=True)

    if args.list_ports:
        return

    if args.preset is not None:
        packet = build_preset_packet(args.preset, args.device_id)
    else:
        packet = parse_hex_string(args.hex)

    if not packet:
        raise ValueError("No packet bytes to send. Provide --preset or --hex.")

    try:
        import serial
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import pyserial. Please run this script with the project environment, "
            "for example `.venv/bin/python`."
        ) from exc

    print(f"Using port: {describe_port(args.port)} baudrate={args.baudrate}", flush=True)
    print(f"TX ({len(packet)} bytes): {to_hex(packet)}", flush=True)

    ser = serial.Serial()
    ser.port = args.port
    ser.baudrate = args.baudrate
    ser.timeout = 0
    ser.open()

    try:
        for attempt in range(args.repeat):
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            ser.write(packet)
            ser.flush()
            recv = read_until_quiet(ser, timeout=args.timeout, quiet_time=args.quiet_time)

            print(f"[attempt {attempt + 1}/{args.repeat}] RX ({len(recv)} bytes): {to_hex(recv)}", flush=True)
            if recv:
                print(f"[attempt {attempt + 1}/{args.repeat}] RX raw: {recv!r}", flush=True)

            if attempt + 1 < args.repeat:
                time.sleep(args.gap)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
