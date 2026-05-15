#!/usr/bin/env python3
import sys
sys.path.insert(0, r'C:\Users\admin\gemma4-turbo-family')
from merge_mmproj import merge_gguf

print("Merging IQ3_M ultra with vision encoder...")
merge_gguf(
    r'C:\Users\admin\gemma4-turbo-family\gemma4-e2b-iq3m-ultra.gguf',
    r'C:\Users\admin\.ollama\models\blobs\sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448',
    r'C:\Users\admin\gemma4-turbo-family\gemma4-e2b-iq3m-ultra-mmproj.gguf'
)
print("Done!")
