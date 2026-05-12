# gemma4-nano Model Card

## Model Description

Ultra-compressed Gemma 4 models (Q3_K_S quantization) optimized for mobile, edge, and resource-constrained environments. 50-57% smaller than stock models with 13% faster inference.

## Intended Use

- Mobile and edge AI applications with limited RAM (<8GB)
- Offline-first apps requiring small downloads
- IoT and embedded systems
- Battery-sensitive devices
- Rapid prototyping and development

## Model Sizes

- **e2b**: 3.1 GB (fits 4GB RAM devices)
- **e4b**: 4.7 GB (recommended default)

## Performance

E2b nano achieves **17.3 tok/s average** (1.13x faster than gemma4-turbo) on CPU with 8 threads.

## Limitations

- Text-only (no vision capability)
- Slightly lower precision than 4-bit quantization on rare edge cases
- Optimized for English (same as base Gemma 4)

## Training Data

Same as Google Gemma 4 base models (not retrained, only quantized).

## Ethical Considerations

Inherits all ethical considerations from base Gemma 4 models. See [Gemma documentation](https://ai.google.dev/gemma/docs/gemma_terms) for responsible AI guidelines.

## Citation

```
@software{gemma4-nano,
  author = {ssfdre38},
  title = {gemma4-nano: Ultra-compressed Gemma 4 for mobile and edge},
  year = {2026},
  url = {https://ollama.com/ssfdre38/gemma4-nano}
}
```

Related: [gemma4-turbo](https://ollama.com/ssfdre38/gemma4-turbo)
