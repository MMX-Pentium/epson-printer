#!/usr/bin/env python3
"""
Carve CROM images out of a mixed binary.

This parser follows the same length logic as uncrom.py/uncrom.c:
  - file starts with "CROM"
  - one CROM contains one or more segments
  - each segment has JPEG-like markers (FFD8, FFC4, FFB1, FFB2)
  - coded_bytes and literal_len determine variable payload sizes
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


MAGIC = b"CROM"
SOI = 0xFFD8
DHT = 0xFFC4
COPY_INFO = 0xFFB1
LITERAL_INFO = 0xFFB2


class ParseError(Exception):
    pass


class TruncatedError(ParseError):
    def __init__(self, what: str, wanted: int, got: int):
        self.what = what
        self.wanted = wanted
        self.got = got
        super().__init__(f"{what}: wanted {wanted} bytes, got {got}")


@dataclass
class SegmentInfo:
    start: int
    end: int
    total_length_field: int
    huff_payload_len: int
    copy_info_payload_len: int
    coded_bytes: int
    num_items: int
    literal_len: int


@dataclass
class CarvedCrom:
    start: int
    end: int
    segments: List[SegmentInfo]


def _need(data: bytes, off: int, count: int, what: str) -> None:
    got = len(data) - off
    if got < count:
        raise TruncatedError(what, count, max(0, got))


def _u16be(data: bytes, off: int) -> int:
    return struct.unpack_from(">H", data, off)[0]


def _u32be(data: bytes, off: int) -> int:
    return struct.unpack_from(">I", data, off)[0]


def _take_marker(data: bytes, off: int, want_mark: int) -> Tuple[memoryview, int]:
    _need(data, off, 4, f"marker 0x{want_mark:04x} header")
    mark = _u16be(data, off)
    if mark != want_mark:
        raise ParseError(f"expected marker 0x{want_mark:04x}, got 0x{mark:04x} at 0x{off:x}")

    length = _u16be(data, off + 2)
    if length < 2:
        raise ParseError(f"invalid marker length {length} for 0x{want_mark:04x} at 0x{off:x}")

    payload_len = length - 2
    payload_start = off + 4
    _need(data, payload_start, payload_len, f"marker 0x{want_mark:04x} payload")
    payload_end = payload_start + payload_len
    return memoryview(data)[payload_start:payload_end], payload_end


def parse_segment(data: bytes, off: int) -> SegmentInfo:
    seg_start = off
    _need(data, off, 6, "segment SOI header")
    soi = _u16be(data, off)
    if soi != SOI:
        raise ParseError(f"expected SOI 0x{SOI:04x}, got 0x{soi:04x} at 0x{off:x}")
    total_length_field = _u32be(data, off + 2)
    off += 6

    huff_payload, off = _take_marker(data, off, DHT)
    copy_info_payload, off = _take_marker(data, off, COPY_INFO)
    if len(copy_info_payload) < 9:
        raise ParseError(
            f"COPY_INFO payload too short ({len(copy_info_payload)}), need >= 9 at 0x{off:x}"
        )

    # Same offsets as uncrom.py/uncrom.c:
    # [1:5] coded_bytes, [5:9] num_items (big endian)
    coded_bytes = struct.unpack_from(">I", copy_info_payload, 1)[0]
    num_items = struct.unpack_from(">I", copy_info_payload, 5)[0]

    _need(data, off, coded_bytes, "compressed copy data")
    off += coded_bytes

    literal_info_payload, off = _take_marker(data, off, LITERAL_INFO)
    if len(literal_info_payload) < 4:
        raise ParseError(
            f"LITERAL_INFO payload too short ({len(literal_info_payload)}), need >= 4 at 0x{off:x}"
        )
    literal_len = struct.unpack_from(">I", literal_info_payload, 0)[0]

    _need(data, off, literal_len, "literal data")
    off += literal_len

    return SegmentInfo(
        start=seg_start,
        end=off,
        total_length_field=total_length_field,
        huff_payload_len=len(huff_payload),
        copy_info_payload_len=len(copy_info_payload),
        coded_bytes=coded_bytes,
        num_items=num_items,
        literal_len=literal_len,
    )


def parse_crom_at(data: bytes, start: int, min_segments: int = 1) -> Optional[CarvedCrom]:
    if data[start : start + 4] != MAGIC:
        return None

    off = start + 4
    segments: List[SegmentInfo] = []

    while True:
        if off + 6 > len(data):
            break

        # uncrom.py treats all-FF bytes at segment boundary as end-of-CROM padding.
        if data[off : off + 6] == b"\xff" * 6:
            break

        try:
            seg = parse_segment(data, off)
        except ParseError:
            break

        segments.append(seg)
        off = seg.end

    if len(segments) < min_segments:
        return None

    return CarvedCrom(start=start, end=off, segments=segments)


def carve_all(data: bytes, min_segments: int = 1) -> List[CarvedCrom]:
    results: List[CarvedCrom] = []
    pos = 0

    while True:
        idx = data.find(MAGIC, pos)
        if idx < 0:
            break

        carved = parse_crom_at(data, idx, min_segments=min_segments)
        if carved is not None:
            results.append(carved)
            # Skip ahead to end of this carved region to avoid nested matches.
            pos = max(carved.end, idx + 1)
        else:
            pos = idx + 1

    return results


def default_prefix(input_path: Path) -> str:
    return input_path.stem + ".carved"


def write_outputs(
    data: bytes,
    carved_list: List[CarvedCrom],
    outdir: Path,
    prefix: str,
    dry_run: bool = False,
) -> List[dict]:
    meta = []

    for i, carved in enumerate(carved_list):
        out_name = f"{prefix}.{i:03d}.crom"
        out_path = outdir / out_name
        info = {
            "index": i,
            "offset_start": carved.start,
            "offset_end": carved.end,
            "size": carved.end - carved.start,
            "segments": len(carved.segments),
            "output": str(out_path),
            "segment_info": [
                {
                    "start": s.start,
                    "end": s.end,
                    "size": s.end - s.start,
                    "total_length_field": s.total_length_field,
                    "huff_payload_len": s.huff_payload_len,
                    "copy_info_payload_len": s.copy_info_payload_len,
                    "coded_bytes": s.coded_bytes,
                    "num_items": s.num_items,
                    "literal_len": s.literal_len,
                }
                for s in carved.segments
            ],
        }
        meta.append(info)

        if not dry_run:
            out_path.write_bytes(data[carved.start : carved.end])

    return meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Carve CROM images from a mixed binary using uncrom-compatible length parsing."
    )
    parser.add_argument("input", help="input binary file")
    parser.add_argument(
        "-o", "--outdir", default=".", help="output directory (default: current directory)"
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="output filename prefix (default: <input_stem>.carved)",
    )
    parser.add_argument(
        "--min-segments",
        type=int,
        default=1,
        help="minimum valid segments required per carved CROM (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and report only; do not write .crom files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON summary",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    prefix = args.prefix or default_prefix(input_path)

    data = input_path.read_bytes()
    carved_list = carve_all(data, min_segments=args.min_segments)

    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    meta = write_outputs(data, carved_list, outdir, prefix, dry_run=args.dry_run)

    if args.json:
        print(json.dumps(meta, indent=2))
    else:
        print(f"input: {input_path} ({len(data)} bytes)")
        print(f"found: {len(meta)} CROM image(s)")
        for item in meta:
            print(
                "  #{index:03d}: off=0x{offset_start:x}-0x{offset_end:x} "
                "size={size} segments={segments} -> {output}".format(**item)
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
