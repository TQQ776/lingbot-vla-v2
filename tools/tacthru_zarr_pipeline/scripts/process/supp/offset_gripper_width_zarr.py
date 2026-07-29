from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import click
import numpy as np
import zarr


GRIPPER_PREFIX = "data/robot0_gripper_width/"


def _build_replacement_dir(src: Path, *, offset: float, tmp_root: Path) -> tuple[Path, np.ndarray]:
    out_dir = tmp_root / "replacement"
    store = zarr.DirectoryStore(str(out_dir))

    with zarr.ZipStore(str(src), mode="r") as src_store:
        src_root = zarr.group(store=src_store)
        src_arr = src_root["data"]["robot0_gripper_width"]
        src_data = np.asarray(src_arr[:], dtype=np.float32)
        dst_data = np.maximum(src_data - np.float32(offset), 0.0)

        data_group = zarr.group(store=store).require_group("data")
        dst_arr = data_group.create_dataset(
            "robot0_gripper_width",
            shape=src_arr.shape,
            chunks=src_arr.chunks,
            dtype=src_arr.dtype,
            compressor=src_arr.compressor,
            fill_value=src_arr.fill_value,
            order=src_arr.order,
            overwrite=True,
        )
        dst_arr[:] = dst_data

    return out_dir, dst_data.reshape(-1)


def _clone_info(src_info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    dst_info = zipfile.ZipInfo(filename=src_info.filename, date_time=src_info.date_time)
    dst_info.compress_type = src_info.compress_type
    dst_info.comment = src_info.comment
    dst_info.extra = src_info.extra
    dst_info.create_system = src_info.create_system
    dst_info.create_version = src_info.create_version
    dst_info.extract_version = src_info.extract_version
    dst_info.flag_bits = src_info.flag_bits
    dst_info.volume = src_info.volume
    dst_info.internal_attr = src_info.internal_attr
    dst_info.external_attr = src_info.external_attr
    return dst_info


def _rewrite_zip(src: Path, dst: Path, replacement_dir: Path) -> tuple[int, int]:
    copied = 0
    skipped = 0
    replacement_files = sorted(path for path in replacement_dir.rglob("*") if path.is_file())

    with zipfile.ZipFile(src, mode="r") as src_zip, zipfile.ZipFile(
        dst, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as dst_zip:
        for idx, src_info in enumerate(src_zip.infolist(), start=1):
            if src_info.filename.startswith(GRIPPER_PREFIX):
                skipped += 1
                continue

            dst_info = _clone_info(src_info)
            with src_zip.open(src_info, mode="r") as src_fp, dst_zip.open(
                dst_info, mode="w", force_zip64=True
            ) as dst_fp:
                shutil.copyfileobj(src_fp, dst_fp, length=1024 * 1024)
            copied += 1

            if idx % 50000 == 0:
                print(f"rewrote {idx} / {len(src_zip.infolist())} zip entries", flush=True)

        for path in replacement_files:
            arcname = path.relative_to(replacement_dir).as_posix()
            if arcname in {".zgroup", "data/.zgroup"}:
                continue
            dst_info = zipfile.ZipInfo(filename=arcname)
            dst_info.compress_type = zipfile.ZIP_STORED
            with path.open("rb") as src_fp, dst_zip.open(dst_info, mode="w", force_zip64=True) as dst_fp:
                shutil.copyfileobj(src_fp, dst_fp, length=1024 * 1024)

    return copied, skipped


@click.command(help="Create a new zarr.zip dataset with robot0_gripper_width shifted down by a fixed offset.")
@click.argument("src", type=click.Path(path_type=Path, exists=True))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option("--offset", type=float, required=True)
def main(src: Path, dst: Path, offset: float) -> None:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()

    print(f"source: {src}", flush=True)
    print(f"target: {dst}", flush=True)
    print(f"offset: {offset}", flush=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    with tempfile.TemporaryDirectory(prefix="gripper_width_fix_", dir=str(dst.parent)) as tmp:
        tmp_root = Path(tmp)
        replacement_dir, adjusted = _build_replacement_dir(src, offset=offset, tmp_root=tmp_root)
        print(
            "replacement stats:",
            f"min={float(adjusted.min()):.9f}",
            f"max={float(adjusted.max()):.9f}",
            f"mean={float(adjusted.mean()):.9f}",
            flush=True,
        )

        copied, skipped = _rewrite_zip(src, dst, replacement_dir)
        print(f"zip rewritten: copied={copied} skipped_old_gripper_entries={skipped}", flush=True)

    with zarr.ZipStore(str(dst), mode="r") as verify_store:
        root = zarr.group(store=verify_store)
        arr = np.asarray(root["data"]["robot0_gripper_width"][:], dtype=np.float32).reshape(-1)
        q = np.quantile(arr, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
        print(f"verify min={float(arr.min()):.9f}", flush=True)
        print(f"verify max={float(arr.max()):.9f}", flush=True)
        print(f"verify mean={float(arr.mean()):.9f}", flush=True)
        print(f"verify quantiles={np.array2string(q, precision=9, suppress_small=False)}", flush=True)


if __name__ == "__main__":
    main()
