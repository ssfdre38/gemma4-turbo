#!/usr/bin/env python3
"""
Zero-dependency streaming GGUF v3 multimodal merger for Gemma 4.5 Turbo.
Combines a baked IQ4_XS language model GGUF with the vision/audio encoder
tensors (a.*, v.*, mm.*) and metadata (gemma4.vision.*, gemma4.audio.*, clip.*)
from an existing multimodal GGUF in 16 MiB streaming chunks (<32 MiB RAM).
"""

import os
import struct
import sys

GGUF_MAGIC = 0x46554747
VISION_TENSOR_PREFIXES = ("a.", "v.", "mm.")
VISION_KV_PREFIXES = ("gemma4.vision.", "gemma4.audio.", "clip.")

BLOCK_INFO = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 36),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    19: (256, 110), # IQ4_XS legacy/alias guard
    20: (32, 18),   # IQ4_NL
    23: (256, 136), # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    30: (1, 2),     # BF16
}

SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def align_to(offset: int, alignment: int) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def read_raw_value(f, vtype: int) -> bytes:
    if vtype in SCALAR_SIZES:
        return f.read(SCALAR_SIZES[vtype])
    if vtype == 8:
        lb = f.read(8)
        n = struct.unpack("<Q", lb)[0]
        return lb + f.read(n)
    if vtype == 9:
        hdr = f.read(12)
        at, al = struct.unpack("<IQ", hdr)
        if at in SCALAR_SIZES:
            return hdr + f.read(al * SCALAR_SIZES[at])
        parts = [hdr]
        for _ in range(al):
            parts.append(read_raw_value(f, at))
        return b"".join(parts)
    raise ValueError(f"Unsupported GGUF KV value type: {vtype}")


def parse_gguf(path: str):
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path}")

        kvs = []
        alignment = 32
        for _ in range(n_kv):
            kl = struct.unpack("<Q", f.read(8))[0]
            key = f.read(kl).decode("utf-8")
            vt = struct.unpack("<I", f.read(4))[0]
            raw_v = read_raw_value(f, vt)
            if key == "general.alignment" and vt == 4 and len(raw_v) == 4:
                al = struct.unpack("<I", raw_v)[0]
                if al >= 1 and (al & (al - 1)) == 0:
                    alignment = al
            kvs.append((key, vt, raw_v))

        tensors = []
        for _ in range(n_tensors):
            nl = struct.unpack("<Q", f.read(8))[0]
            name = f.read(nl).decode("utf-8")
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tt, rel_off = struct.unpack("<IQ", f.read(12))
            bs, bb = BLOCK_INFO[tt]
            rows = 1
            for d in dims[1:]:
                rows *= d
            nbytes = (dims[0] // bs) * bb * rows
            tensors.append({
                "name": name,
                "dims": dims,
                "type": tt,
                "rel_off": rel_off,
                "nbytes": nbytes,
                "src": path,
            })

        data_start = align_to(f.tell(), alignment)
        for t in tensors:
            t["abs_off"] = data_start + t["rel_off"]

    return version, alignment, kvs, tensors


def stream_copy(src_f, abs_off: int, nbytes: int, dst_f, chunk_size: int = 16 * 1024 * 1024):
    src_f.seek(abs_off)
    rem = nbytes
    while rem > 0:
        n = min(rem, chunk_size)
        buf = src_f.read(n)
        if len(buf) != n:
            raise IOError(f"Short read at offset {abs_off}: expected {n}, got {len(buf)}")
        dst_f.write(buf)
        rem -= n


def merge_mmproj(lm_path: str, mm_src_path: str, out_path: str):
    print(f"Merging:\n  LM : {lm_path}\n  MM : {mm_src_path}\n  OUT: {out_path}", flush=True)
    ver, alignment, lm_kvs, lm_tensors = parse_gguf(lm_path)
    _, _, mm_kvs, mm_tensors = parse_gguf(mm_src_path)

    lm_keys = {k for k, _, _ in lm_kvs}
    extra_kvs = [
        (k, vt, rv) for k, vt, rv in mm_kvs
        if k not in lm_keys and k.startswith(VISION_KV_PREFIXES)
    ]
    all_kvs = lm_kvs + extra_kvs

    lm_names = {t["name"] for t in lm_tensors}
    vision_tensors = [
        t for t in mm_tensors
        if t["name"] not in lm_names and t["name"].startswith(VISION_TENSOR_PREFIXES)
    ]
    all_tensors = lm_tensors + vision_tensors
    print(f"  KVs: {len(lm_kvs)} LM + {len(extra_kvs)} MM = {len(all_kvs)}", flush=True)
    print(f"  Tensors: {len(lm_tensors)} LM + {len(vision_tensors)} MM = {len(all_tensors)}", flush=True)

    out_offsets = []
    cur = 0
    for t in all_tensors:
        out_offsets.append(cur)
        cur = align_to(cur + t["nbytes"], alignment)

    with open(lm_path, "rb") as f_lm, open(mm_src_path, "rb") as f_mm, open(out_path, "wb") as f_out:
        f_out.write(struct.pack("<IIQQ", GGUF_MAGIC, ver, len(all_tensors), len(all_kvs)))
        for k, vt, rv in all_kvs:
            kb = k.encode("utf-8")
            f_out.write(struct.pack("<Q", len(kb)))
            f_out.write(kb)
            f_out.write(struct.pack("<I", vt))
            f_out.write(rv)

        for i, t in enumerate(all_tensors):
            nb = t["name"].encode("utf-8")
            f_out.write(struct.pack("<Q", len(nb)))
            f_out.write(nb)
            f_out.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                f_out.write(struct.pack("<Q", d))
            f_out.write(struct.pack("<IQ", t["type"], out_offsets[i]))

        pos = f_out.tell()
        pad = align_to(pos, alignment) - pos
        if pad:
            f_out.write(b"\x00" * pad)

        for t in all_tensors:
            src_f = f_lm if t["src"] == lm_path else f_mm
            stream_copy(src_f, t["abs_off"], t["nbytes"], f_out)
            rem_pad = align_to(t["nbytes"], alignment) - t["nbytes"]
            if rem_pad:
                f_out.write(b"\x00" * rem_pad)

    sz_gb = os.path.getsize(out_path) / (1024 ** 3)
    print(f"  ✓ Wrote {out_path} ({sz_gb:.2f} GiB)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <lm.gguf> <mm_source.gguf> <out.gguf>")
        sys.exit(1)
    merge_mmproj(sys.argv[1], sys.argv[2], sys.argv[3])
