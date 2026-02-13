#!/usr/bin/env python3
"""One-off converter for legacy V4DW world files.

Converts V4DW version 1 files to version 2 and sets a base-world mode.
Default base mode is flat-floor (material 11) so legacy flat worlds gain
infinite horizon behavior after conversion.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path
from typing import List, Tuple

MAGIC = b"V4DW"
VERSION_V1 = 1
VERSION_V2 = 2

CHUNK_VOLUME = 8 * 8 * 8 * 8

ENCODING_UNIFORM = 0
ENCODING_RLE = 1
ENCODING_RAW = 2

BASE_KIND_EMPTY = 0
BASE_KIND_FLAT_FLOOR = 1


class ParseError(RuntimeError):
    pass


def _read_u16(data: bytes, off: int) -> Tuple[int, int]:
    if off + 2 > len(data):
        raise ParseError("unexpected EOF while reading u16")
    return struct.unpack_from("<H", data, off)[0], off + 2


def _read_u32(data: bytes, off: int) -> Tuple[int, int]:
    if off + 4 > len(data):
        raise ParseError("unexpected EOF while reading u32")
    return struct.unpack_from("<I", data, off)[0], off + 4


def _read_i32(data: bytes, off: int) -> Tuple[int, int]:
    if off + 4 > len(data):
        raise ParseError("unexpected EOF while reading i32")
    return struct.unpack_from("<i", data, off)[0], off + 4


def _read_chunk_payload(data: bytes, off: int) -> Tuple[bytes, int]:
    if off >= len(data):
        raise ParseError("unexpected EOF while reading chunk encoding")

    start = off
    encoding = data[off]
    off += 1

    if encoding == ENCODING_UNIFORM:
        if off + 1 > len(data):
            raise ParseError("unexpected EOF while reading uniform voxel")
        off += 1
        return data[start:off], off

    if encoding == ENCODING_RAW:
        if off + CHUNK_VOLUME > len(data):
            raise ParseError("unexpected EOF while reading raw chunk")
        off += CHUNK_VOLUME
        return data[start:off], off

    if encoding == ENCODING_RLE:
        run_count, off = _read_u16(data, off)
        expanded = 0
        for _ in range(run_count):
            if off + 1 > len(data):
                raise ParseError("unexpected EOF while reading RLE voxel")
            off += 1
            run_len, off = _read_u16(data, off)
            expanded += run_len
            if expanded > CHUNK_VOLUME:
                raise ParseError("RLE overflow while parsing chunk")
        if expanded != CHUNK_VOLUME:
            raise ParseError(
                f"RLE chunk expanded to {expanded} voxels (expected {CHUNK_VOLUME})"
            )
        return data[start:off], off

    raise ParseError(f"unknown chunk encoding {encoding}")


def parse_v1(data: bytes) -> List[Tuple[int, int, int, int, bytes]]:
    if len(data) < 8:
        raise ParseError("file too small")
    if data[:4] != MAGIC:
        raise ParseError("invalid magic")

    version = struct.unpack_from("<I", data, 4)[0]
    if version != VERSION_V1:
        raise ParseError(f"expected V4DW version 1, found version {version}")

    off = 8
    chunk_count, off = _read_u32(data, off)

    chunks: List[Tuple[int, int, int, int, bytes]] = []
    for _ in range(chunk_count):
        x, off = _read_i32(data, off)
        y, off = _read_i32(data, off)
        z, off = _read_i32(data, off)
        w, off = _read_i32(data, off)
        payload, off = _read_chunk_payload(data, off)
        chunks.append((x, y, z, w, payload))

    if off != len(data):
        trailing = len(data) - off
        raise ParseError(f"unexpected trailing bytes: {trailing}")

    return chunks


def build_v2(
    chunks: List[Tuple[int, int, int, int, bytes]],
    base_kind: int,
    floor_material: int,
) -> bytes:
    payload_index_by_bytes = {}
    payloads: List[bytes] = []
    entries: List[Tuple[int, int, int, int, int]] = []

    for x, y, z, w, payload in chunks:
        payload_idx = payload_index_by_bytes.get(payload)
        if payload_idx is None:
            payload_idx = len(payloads)
            payload_index_by_bytes[payload] = payload_idx
            payloads.append(payload)
        entries.append((x, y, z, w, payload_idx))

    out = bytearray()
    out.extend(MAGIC)
    out.extend(struct.pack("<I", VERSION_V2))

    out.append(base_kind)
    if base_kind == BASE_KIND_FLAT_FLOOR:
        out.append(floor_material)

    out.extend(struct.pack("<I", len(entries)))
    out.extend(struct.pack("<I", len(payloads)))

    for payload in payloads:
        out.extend(payload)

    for x, y, z, w, payload_idx in entries:
        out.extend(struct.pack("<iiiiI", x, y, z, w, payload_idx))

    return bytes(out)


def convert_file(
    input_path: Path,
    output_path: Path,
    base_kind: int,
    floor_material: int,
    in_place: bool,
    backup_suffix: str,
) -> None:
    raw = input_path.read_bytes()
    chunks = parse_v1(raw)
    upgraded = build_v2(chunks, base_kind, floor_material)

    if in_place:
        backup_path = Path(str(input_path) + backup_suffix)
        if backup_path.exists():
            raise RuntimeError(
                f"backup already exists: {backup_path}; remove it or choose a different suffix"
            )
        shutil.copy2(input_path, backup_path)

        tmp_path = Path(str(input_path) + ".tmp")
        tmp_path.write_bytes(upgraded)
        tmp_path.replace(input_path)
        print(f"Wrote upgraded file in place: {input_path}")
        print(f"Backup created: {backup_path}")
    else:
        output_path.write_bytes(upgraded)
        print(f"Wrote upgraded file: {output_path}")

    payload_count = len({payload for *_, payload in chunks})
    print(
        "Converted chunks: "
        f"{len(chunks)} (unique payloads: {payload_count})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert V4DW v1 world file to v2 with configurable base world"
    )
    parser.add_argument("input", type=Path, help="Path to legacy v1 .v4dw file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <input>.v2)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file in place (creates backup)",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".v1.bak",
        help="Backup suffix used with --in-place (default: .v1.bak)",
    )
    parser.add_argument(
        "--base",
        choices=("flat-floor", "empty"),
        default="flat-floor",
        help="Base world mode to write in v2 output (default: flat-floor)",
    )
    parser.add_argument(
        "--floor-material",
        type=int,
        default=11,
        help="Material id for flat-floor base (default: 11)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file does not exist: {args.input}", file=sys.stderr)
        return 2

    if not (0 <= args.floor_material <= 255):
        print("--floor-material must be in [0,255]", file=sys.stderr)
        return 2

    if args.in_place and args.output is not None:
        print("--output cannot be used with --in-place", file=sys.stderr)
        return 2

    output_path = args.output or Path(str(args.input) + ".v2")
    if not args.in_place and output_path == args.input:
        print("Refusing to overwrite input without --in-place", file=sys.stderr)
        return 2

    base_kind = BASE_KIND_FLAT_FLOOR if args.base == "flat-floor" else BASE_KIND_EMPTY

    try:
        convert_file(
            input_path=args.input,
            output_path=output_path,
            base_kind=base_kind,
            floor_material=args.floor_material,
            in_place=args.in_place,
            backup_suffix=args.backup_suffix,
        )
    except (ParseError, RuntimeError, OSError) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
