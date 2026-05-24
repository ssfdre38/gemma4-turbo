from mom_slice_extract import merge_slice, eval_model
import os

if __name__ == "__main__":
    anchor = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-q3ks-nano.gguf'
    slice_f = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-q2k-deswa.gguf'
    
    print("Precise Cliff Detection (Layers 25-30)...")
    for i in range(25, 31):
        # We test a block from i to 34. 
        # This tells us: "How early can we start the Q2_K slice?"
        target_layers = list(range(i, 35))
        test_file = f"cliff_test_{i}.gguf"
        print(f"  Testing Q2_K from Layer {i} to 34... ", end='', flush=True)
        merge_slice(anchor, slice_f, test_file, target_layers)
        res = eval_model(test_file, f"cliff-test-{i}")
        print(f"Result: [{res}]")
        
        if os.path.exists(test_file): os.remove(test_file)
        os.system(f"ollama rm cliff-test-{i} > nul 2>&1")
