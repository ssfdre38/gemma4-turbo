import sys
import struct
import os
from pathlib import Path

try:
    import gguf
except ImportError:
    print("ERROR: pip install gguf")
    sys.exit(1)

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

def align_to(offset, alignment):
    return ((offset + alignment - 1) // alignment) * alignment

def raw_kv_bytes(field):
    return b''.join(part.tobytes() for part in field.parts)

def tensor_info_bytes(tensor, data_offset):
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

def surgical_merge(anchor_path, slice_path, output_path, anchor_layers):
    """
    anchor_path: Path to the high-precision GGUF (e.g., IQ3_M or Q4)
    slice_path: Path to the low-precision GGUF (e.g., IQ2_M or Q2_K)
    anchor_layers: List of layer indices (ints) to keep from the anchor model.
    """
    print(f"--- MoM Surgical Merge ---")
    print(f"  Anchor (High): {anchor_path}")
    print(f"  Slice  (Low):  {slice_path}")
    print(f"  Anchoring Layers: {anchor_layers}")

    anchor = gguf.GGUFReader(anchor_path)
    slice_mdl = gguf.GGUFReader(slice_path)

    # 1. Use KV metadata from Anchor model
    SKIP_INTERNAL = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}
    all_kv = [(k, v) for k, v in anchor.fields.items() if k not in SKIP_INTERNAL]

    # 2. Select Tensors
    all_tensors = []
    
    # Map slice tensors by name for quick lookup
    slice_tensor_map = {t.name: t for t in slice_mdl.tensors}
    
    for t_anchor in anchor.tensors:
        name = t_anchor.name
        
        # Non-block tensors (embeddings, norms, etc.) always use Anchor precision
        if not name.startswith('blk.'):
            all_tensors.append(t_anchor)
            continue
            
        # Determine layer index
        parts = name.split('.')
        try:
            layer_idx = int(parts[1])
        except (IndexError, ValueError):
            all_tensors.append(t_anchor) # Fallback to anchor
            continue

        if layer_idx in anchor_layers:
            # Use high-precision tensor
            all_tensors.append(t_anchor)
        else:
            # Use low-precision tensor from slice model
            if name in slice_tensor_map:
                all_tensors.append(slice_tensor_map[name])
            else:
                print(f"  WARNING: Tensor {name} missing in slice model, using anchor.")
                all_tensors.append(t_anchor)

    # 3. Calculate Offsets
    alignment = DEFAULT_ALIGNMENT
    if 'general.alignment' in anchor.fields:
        alignment = int(anchor.fields['general.alignment'].parts[-1][0])

    tensor_offsets = []
    current_offset = 0
    for t in all_tensors:
        tensor_offsets.append(current_offset)
        size = int(t.n_bytes)
        current_offset = align_to(current_offset + size, alignment)

    # 4. Write GGUF
    print(f"  Writing: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(all_tensors)))
        f.write(struct.pack('<Q', len(all_kv)))

        for key, field in all_kv:
            f.write(raw_kv_bytes(field))

        for i, t in enumerate(all_tensors):
            f.write(tensor_info_bytes(t, tensor_offsets[i]))

        pos = f.tell()
        aligned = align_to(pos, alignment)
        f.write(b'\x00' * (aligned - pos))

        for i, t in enumerate(all_tensors):
            data = bytes(t.data)
            f.write(data)
            size = len(data)
            padded = align_to(size, alignment)
            if padded > size:
                f.write(b'\x00' * (padded - size))

    print(f"  SUCCESS: {output_path} ({os.path.getsize(output_path)/1e9:.2f} GB)")

if __name__ == "__main__":
    # Final High-Res Data: The Resilient Zone is exactly Layers 30-40.
    # Everything else (0-29 and 41) is CRITICAL.
    anchors = list(range(0, 30)) + [41]
    
    # Paths (adjust based on what we have)
    anchor_file = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-q3ks-nano.gguf'
    slice_file  = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-q2k-deswa.gguf'
    out_file    = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-mom-final.gguf'
    
    if os.path.exists(anchor_file) and os.path.exists(slice_file):
        surgical_merge(anchor_file, slice_file, out_file, anchors)
    else:
        print("Required files for merge not found.")
