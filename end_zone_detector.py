from mom_slice_extract import merge_slice, eval_model
import os

if __name__ == "__main__":
    anchor = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-q3ks-nano.gguf'
    slice_f = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-q2k-deswa.gguf'
    
    print("Precise End-Zone Detection (Layers 35-41)...")
    for i in range(35, 42):
        # We test a block from 30 to i. 
        # This tells us: "How late can we end the Q2_K slice?"
        target_layers = list(range(30, i + 1))
        test_file = f"end_zone_test_{i}.gguf"
        print(f"  Testing Q2_K from Layer 30 to {i}... ", end='', flush=True)
        merge_slice(anchor, slice_f, test_file, target_layers)
        res = eval_model(test_file, f"end-test-{i}")
        print(f"Result: [{res}]")
        
        if os.path.exists(test_file): os.remove(test_file)
        os.system(f"ollama rm end-test-{i} > nul 2>&1")
