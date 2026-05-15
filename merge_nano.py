"""Merge vision encoder into Q3_K_S nano model"""
import sys
sys.path.insert(0, r'C:\Users\admin\gemma4-turbo-family')
from merge_mmproj import merge_gguf

# Q3_K_S nano e2b model
merge_gguf(
    lm_path=r'C:\Users\admin\gemma4-turbo-family\gemma4-e2b-q3ks-nano.gguf',
    orig_path=r'C:\Users\admin\.ollama\models\blobs\sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448',
    output_path=r'C:\Users\admin\gemma4-turbo-family\gemma4-e2b-q3ks-mmproj.gguf'
)
print("\n✓ Nano model merged successfully!")
