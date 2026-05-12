#!/usr/bin/env python3
"""
Merge an IQ4_XS language model GGUF with vision tensors from the original
Ollama gemma4 blob into a single combined GGUF with text+vision+thinking support.

Strategy:
- LM tensors (blk.*, output_norm.*, per_layer_*.*, rope_freqs.*, token_embd.*) → from IQ4_XS GGUF
- Vision tensors (a.*, v.*, mm.*) → from original Ollama blob
- KV metadata: all from IQ4_XS GGUF + gemma4.vision.* / gemma4.audio.* from original blob
"""

import sys
import struct
import os
from pathlib import Path

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

VISION_TENSOR_PREFIXES = ('a.', 'v.', 'mm.')
VISION_KV_PREFIXES = ('gemma4.vision.', 'gemma4.audio.')

def align_to(offset, alignment):
    return ((offset + alignment - 1) // alignment) * alignment

def raw_kv_bytes(field):
    """Return the raw binary bytes of a KV pair by concatenating all parts."""
    return b''.join(part.tobytes() for part in field.parts)

def tensor_info_bytes(tensor, data_offset):
    """Serialize tensor info to GGUF binary format."""
    name_bytes = tensor.name.encode('utf-8')
    buf = struct.pack('<Q', len(name_bytes))
    buf += name_bytes
    shape = [int(d) for d in tensor.shape]
    buf += struct.pack('<I', len(shape))
    for d in shape:
        buf += struct.pack('<Q', d)
    buf += struct.pack('<I', int(tensor.tensor_type))
    buf += struct.pack('<Q', data_offset)
    return buf

def merge_gguf(lm_path, orig_path, output_path):
    try:
        import gguf
    except ImportError:
        print("ERROR: pip install gguf")
        sys.exit(1)

    print(f"  Reading IQ4_XS LM:  {lm_path}")
    lm = gguf.GGUFReader(lm_path)
    print(f"  Reading orig blob:  {orig_path}")
    orig = gguf.GGUFReader(orig_path)

    # === Collect KV fields ===
    SKIP_INTERNAL = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}

    lm_kv = [(k, v) for k, v in lm.fields.items() if k not in SKIP_INTERNAL]

    # Add vision/audio KV fields from original blob that are missing from IQ4_XS
    lm_keys = {k for k, _ in lm_kv}
    extra_kv = [
        (k, v) for k, v in orig.fields.items()
        if k not in SKIP_INTERNAL
        and k not in lm_keys
        and any(k.startswith(p) for p in VISION_KV_PREFIXES)
    ]

    all_kv = lm_kv + extra_kv
    print(f"  LM KV fields: {len(lm_kv)}, adding vision KV: {len(extra_kv)}")

    # === Collect tensors ===
    lm_tensors = list(lm.tensors)  # all 720 LM tensors

    orig_vision_tensors = [
        t for t in orig.tensors
        if any(t.name.startswith(p) for p in VISION_TENSOR_PREFIXES)
    ]
    print(f"  LM tensors: {len(lm_tensors)}, vision tensors: {len(orig_vision_tensors)}")

    all_tensors = lm_tensors + orig_vision_tensors

    # === Calculate tensor data offsets ===
    alignment = DEFAULT_ALIGNMENT
    # Check if LM GGUF specifies a custom alignment
    if 'general.alignment' in lm.fields:
        alignment = int(lm.fields['general.alignment'].parts[-1][0])

    tensor_offsets = []
    current_offset = 0
    for t in all_tensors:
        tensor_offsets.append(current_offset)
        size = int(t.n_bytes)
        current_offset = align_to(current_offset + size, alignment)

    # === Write output GGUF ===
    print(f"  Writing: {output_path}")
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(all_tensors)))
        f.write(struct.pack('<Q', len(all_kv)))

        # KV pairs (raw bytes)
        for key, field in all_kv:
            f.write(raw_kv_bytes(field))

        # Tensor info
        for i, t in enumerate(all_tensors):
            f.write(tensor_info_bytes(t, tensor_offsets[i]))

        # Alignment padding before tensor data
        pos = f.tell()
        aligned = align_to(pos, alignment)
        f.write(b'\x00' * (aligned - pos))

        # Tensor data
        for i, t in enumerate(all_tensors):
            data = bytes(t.data)
            f.write(data)
            size = len(data)
            padded = align_to(size, alignment)
            if padded > size:
                f.write(b'\x00' * (padded - size))

    out_gb = os.path.getsize(output_path) / 1e9
    print(f"  Done: {output_path} ({out_gb:.2f} GB)")
    return output_path


MODELS = [
    {
        'name': 'e2b',
        'lm':   r'C:\Users\admin\gemma4-turbo-family\gemma4-e2b-iq4xs-turbo.gguf',
        'orig': r'C:\Users\admin\.ollama\models\blobs\sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448',
        'out':  r'C:\Users\admin\gemma4-turbo-family\gemma4-e2b-iq4xs-mmproj.gguf',
    },
    {
        'name': 'e4b',
        'lm':   r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-iq4xs-turbo.gguf',
        'orig': r'C:\Users\admin\.ollama\models\blobs\sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a',
        'out':  r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-iq4xs-mmproj.gguf',
    },
    {
        'name': '26b',
        'lm':   r'C:\Users\admin\gemma4-turbo-family\gemma4-26b-iq4xs-turbo.gguf',
        'orig': r'C:\Users\admin\.ollama\models\blobs\sha256-7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df',
        'out':  r'C:\Users\admin\gemma4-turbo-family\gemma4-26b-iq4xs-mmproj.gguf',
    },
    {
        'name': '31b',
        'lm':   r'C:\Users\admin\gemma4-turbo-family\gemma4-31b-iq4xs-turbo.gguf',
        'orig': r'C:\Users\admin\.ollama\models\blobs\sha256-280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313',
        'out':  r'C:\Users\admin\gemma4-turbo-family\gemma4-31b-iq4xs-mmproj.gguf',
    },
]


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else [m['name'] for m in MODELS]
    for m in MODELS:
        if m['name'] not in targets:
            continue
        print(f"\n=== Merging {m['name']} ===")
        merge_gguf(m['lm'], m['orig'], m['out'])
    print("\nAll done!")
