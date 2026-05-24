import sys
import struct
import os
import requests
import time
from pathlib import Path

try:
    import gguf
except ImportError:
    print("ERROR: pip install gguf")
    sys.exit(1)

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32
OLLAMA_API = "http://127.0.0.1:11434/api/generate"

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

def merge_slice(anchor_path, slice_path, output_path, target_layers):
    """
    Creates a GGUF where target_layers are from slice_path (low prec), 
    and all others are from anchor_path (high prec).
    """
    anchor = gguf.GGUFReader(anchor_path)
    slice_mdl = gguf.GGUFReader(slice_path)
    slice_tensor_map = {t.name: t for t in slice_mdl.tensors}
    
    SKIP_INTERNAL = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}
    all_kv = [(k, v) for k, v in anchor.fields.items() if k not in SKIP_INTERNAL]
    
    all_tensors = []
    for t_anchor in anchor.tensors:
        name = t_anchor.name
        if not name.startswith('blk.'):
            all_tensors.append(t_anchor)
            continue
            
        layer_idx = int(name.split('.')[1])
        if layer_idx in target_layers:
            all_tensors.append(slice_tensor_map[name])
        else:
            all_tensors.append(t_anchor)

    alignment = DEFAULT_ALIGNMENT
    if 'general.alignment' in anchor.fields:
        alignment = int(anchor.fields['general.alignment'].parts[-1][0])

    tensor_offsets = []
    current_offset = 0
    for t in all_tensors:
        tensor_offsets.append(current_offset)
        size = int(t.n_bytes)
        current_offset = align_to(current_offset + size, alignment)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(all_tensors)))
        f.write(struct.pack('<Q', len(all_kv)))
        for key, field in all_kv: f.write(raw_kv_bytes(field))
        for i, t in enumerate(all_tensors): f.write(tensor_info_bytes(t, tensor_offsets[i]))
        pos = f.tell()
        aligned = align_to(pos, alignment)
        f.write(b'\x00' * (aligned - pos))
        for i, t in enumerate(all_tensors):
            data = bytes(t.data)
            f.write(data)
            size = len(data)
            padded = align_to(size, alignment)
            if padded > size: f.write(b'\x00' * (padded - size))

def eval_model(model_path, model_name):
    # Register in Ollama
    modelfile = f"FROM {model_path}\nPARAMETER temperature 0.0"
    with open("TempModelfile", "w") as f: f.write(modelfile)
    
    os.system(f"ollama create {model_name} -f TempModelfile > nul 2>&1")
    
    # Test reasoning
    prompt = "If a train travels 120 miles in 2 hours, what is its average speed? Respond ONLY with the number."
    payload = {"model": model_name, "prompt": prompt, "stream": False, "options": {"num_predict": 10}}
    
    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=30)
        if resp.status_code == 200:
            answer = resp.json().get('response', '').strip()
            return answer
    except:
        return "ERROR"
    return "TIMEOUT"

def run_high_res_analysis(anchor_file, slice_file):
    print(f"Starting High-Res Sensitivity Analysis (Layers 20-41)...")
    results = []
    
    # Test individual layers
    for i in range(20, 42):
        target_layers = [i]
        
        test_file = f"temp_eval_L{i}.gguf"
        test_model = f"mom-test-L{i}"
        
        print(f"  Testing Layer {i}... ", end='', flush=True)
        merge_slice(anchor_file, slice_file, test_file, target_layers)
        
        result = eval_model(test_file, test_model)
        print(f"Result: [{result}]")
        results.append((i, result))
        
        # Cleanup
        if os.path.exists(test_file): os.remove(test_file)
        os.system(f"ollama rm {test_model} > nul 2>&1")

    print("\n--- HIGH-RES SENSITIVITY MAP ---")
    for i, res in results:
        status = "CRITICAL" if "60" not in res else "RESILIENT"
        print(f"Layer {i}: {res} -> {status}")

if __name__ == "__main__":
    anchor = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-q3ks-nano.gguf'
    slice_f = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-q2k-deswa.gguf'
    if os.path.exists(anchor) and os.path.exists(slice_f):
        run_high_res_analysis(anchor, slice_f)
    else:
        print("Missing required GGUF files.")
