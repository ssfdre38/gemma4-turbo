import os
from mom_slice_extract import merge_slice, eval_model

if __name__ == "__main__":
    anchor = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-q3ks-nano.gguf'
    slice_f = r'C:\Users\admin\gemma4-turbo-family\gemma4-e4b-q2k-deswa.gguf'
    
    # Test ONLY 30-34
    print("Verifying block 30-34 specifically...")
    test_file = "verify_30_34.gguf"
    merge_slice(anchor, slice_f, test_file, list(range(30, 35)))
    res = eval_model(test_file, "verify-30-34")
    print(f"Block 30-34 Result: [{res}]")
    
    # Test ONLY 25-29
    print("Verifying block 25-29 specifically...")
    test_file2 = "verify_25_29.gguf"
    merge_slice(anchor, slice_f, test_file2, list(range(25, 30)))
    res2 = eval_model(test_file2, "verify-25-29")
    print(f"Block 25-29 Result: [{res2}]")
