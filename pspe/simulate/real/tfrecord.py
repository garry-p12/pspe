"""Minimal TFRecord + tf.Example reader, in numpy only.

Real wildfire and flood datasets ship as TFRecord (the Next Day Wildfire Spread
release among them), and the obvious way to read them is TensorFlow. That is a
~600 MB dependency, pulled in to parse a container format that is roughly forty
lines of framing, and it is the single most fragile install on Apple silicon and
aarch64 — the two platforms this repository actually runs on.

So: read the format directly.

TFRecord framing, per record:

    uint64  length              (little endian)
    uint32  masked crc32c of the 8 length bytes
    bytes   payload             (a serialised protobuf, here tf.Example)
    uint32  masked crc32c of the payload

`tf.Example` is a protobuf; the subset needed here is a map from feature name to
one of {bytes_list, float_list, int64_list}. Geospatial rasters arrive as
float_list, one flattened HxW array per feature.

The writer exists so tests can build genuine fixtures — CRCs and all — rather
than asserting the reader against files only the reader could produce.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

import numpy as np

# --------------------------------------------------------------------------- #
# CRC32C (Castagnoli), as TFRecord uses it
# --------------------------------------------------------------------------- #
_CRC_POLY = 0x82F63B78
_CRC_TABLE: list[int] = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ (_CRC_POLY if _c & 1 else 0)
    _CRC_TABLE.append(_c)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc ^ 0xFFFFFFFF


def mask_crc(crc: int) -> int:
    """TFRecord stores a rotated CRC, not the raw one."""
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


# --------------------------------------------------------------------------- #
# Protobuf wire format — only what tf.Example needs
# --------------------------------------------------------------------------- #
def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes) -> Iterator[tuple[int, int, bytes | int]]:
    """Yield (field_number, wire_type, value) for one protobuf message."""
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 0x07
        if wire == 0:                       # varint
            value, pos = _read_varint(buf, pos)
            yield field, wire, value
        elif wire == 1:                     # 64-bit
            yield field, wire, buf[pos:pos + 8]
            pos += 8
        elif wire == 2:                     # length-delimited
            length, pos = _read_varint(buf, pos)
            yield field, wire, buf[pos:pos + length]
            pos += length
        elif wire == 5:                     # 32-bit
            yield field, wire, buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _parse_feature(buf: bytes) -> np.ndarray | list[bytes]:
    """One `Feature`: bytes_list (1), float_list (2), or int64_list (3)."""
    for field, _wire, value in _fields(buf):
        assert isinstance(value, bytes)
        if field == 1:                                          # BytesList
            return [v for _f, _w, v in _fields(value) if isinstance(v, bytes)]
        if field == 2:                                          # FloatList
            floats: list[float] = []
            for _f, wire, v in _fields(value):
                if wire == 2 and isinstance(v, bytes):          # packed
                    floats.extend(struct.unpack(f"<{len(v) // 4}f", v))
                elif isinstance(v, bytes):                      # unpacked
                    floats.append(struct.unpack("<f", v)[0])
            return np.asarray(floats, dtype=np.float32)
        if field == 3:                                          # Int64List
            ints: list[int] = []
            for _f, wire, v in _fields(value):
                if wire == 2 and isinstance(v, bytes):
                    pos = 0
                    while pos < len(v):
                        n, pos = _read_varint(v, pos)
                        ints.append(n)
                elif isinstance(v, int):
                    ints.append(v)
            return np.asarray(ints, dtype=np.int64)
    return np.asarray([], dtype=np.float32)


def parse_example(payload: bytes) -> dict[str, np.ndarray | list[bytes]]:
    """A serialised `tf.Example` -> {feature name: values}."""
    out: dict[str, np.ndarray | list[bytes]] = {}
    for field, _wire, features in _fields(payload):
        if field != 1 or not isinstance(features, bytes):
            continue
        for ffield, _fwire, entry in _fields(features):         # Features.feature map
            if ffield != 1 or not isinstance(entry, bytes):
                continue
            key: str | None = None
            value: np.ndarray | list[bytes] | None = None
            for efield, _ewire, item in _fields(entry):          # MapEntry
                if efield == 1 and isinstance(item, bytes):
                    key = item.decode("utf-8")
                elif efield == 2 and isinstance(item, bytes):
                    value = _parse_feature(item)
            if key is not None and value is not None:
                out[key] = value
    return out


# --------------------------------------------------------------------------- #
# Record framing
# --------------------------------------------------------------------------- #
def read_records(path: str | Path, verify: bool = False) -> Iterator[bytes]:
    """Yield raw payloads from a TFRecord file.

    `verify` checks the CRCs. Off by default: on a multi-GB file the pure-Python
    CRC costs far more than the parse itself, and a corrupt record surfaces as a
    protobuf error anyway. Tests turn it on.
    """
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            length_crc = struct.unpack("<I", handle.read(4))[0]
            if verify and mask_crc(crc32c(header)) != length_crc:
                raise ValueError(f"{path}: length CRC mismatch — file is corrupt")
            payload = handle.read(length)
            payload_crc = struct.unpack("<I", handle.read(4))[0]
            if verify and mask_crc(crc32c(payload)) != payload_crc:
                raise ValueError(f"{path}: payload CRC mismatch — file is corrupt")
            yield payload


def iter_examples(path: str | Path, verify: bool = False):
    for payload in read_records(path, verify=verify):
        yield parse_example(payload)


# --------------------------------------------------------------------------- #
# Writer — for test fixtures
# --------------------------------------------------------------------------- #
def _feature_bytes(values: np.ndarray) -> bytes:
    """Serialise one `Feature` message holding a packed float_list."""
    packed = struct.pack(f"<{values.size}f", *values.astype(np.float32).ravel())
    float_list = b"\x0a" + _varint(len(packed)) + packed     # FloatList.value, packed
    return b"\x12" + _varint(len(float_list)) + float_list   # Feature.float_list


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def write_example(features: dict[str, np.ndarray]) -> bytes:
    """Serialise {name: float array} as a tf.Example payload."""
    entries = b""
    for key, values in features.items():
        key_bytes = key.encode("utf-8")
        feature = _feature_bytes(values)
        # map<string, Feature> is repeated MapEntry{key = 1, value = 2}, so the
        # Feature message needs its own tag+length as field 2 of the entry.
        entry = (b"\x0a" + _varint(len(key_bytes)) + key_bytes
                 + b"\x12" + _varint(len(feature)) + feature)
        entries += b"\x0a" + _varint(len(entry)) + entry     # Features.feature
    return b"\x0a" + _varint(len(entries)) + entries         # Example.features


def write_records(path: str | Path, payloads: list[bytes]) -> None:
    with open(path, "wb") as handle:
        for payload in payloads:
            header = struct.pack("<Q", len(payload))
            handle.write(header)
            handle.write(struct.pack("<I", mask_crc(crc32c(header))))
            handle.write(payload)
            handle.write(struct.pack("<I", mask_crc(crc32c(payload))))
