# LLM Evaluation Pipeline

Offline 3-metric evaluation pipeline for LLM outputs built for Media Services use cases.
No API dependency — runs entirely on-device using sentence-transformers.

## Metrics
- **Faithfulness** — detects hallucination via cosine similarity between response and retrieved context
- **Relevancy** — detects intent mismatch between query and response  
- **Coherence** — measures logical flow across sentences (length-aware threshold)

## Results
- 8/10 test cases pass all three metrics
- Correctly flags 1 hallucination (Test 09 — Sony vs Dolby Labs)
- Correctly flags 1 intent mismatch (Test 10 — what vs how-to)
- All results logged to CSV with timestamps for regression tracking

## Stack
Python · NumPy · sentence-transformers (all-MiniLM-L6-v2) · CSV logging

## Performance
- **Batch upgrade**: replaced sequential encoding with batched 
  matrix operations — benchmarked 2.8x faster on 10 test cases
- Sequential: 3.15s → Batch: 1.13s on Apple Silicon MacBook Air
- Scales further at higher volumes (7x speedup at 100 responses)

## Evaluation Architecture

### Layer 1 — Embedding-based scoring (`eval_pipeline.py`)
- Faithfulness, relevancy, coherence via cosine similarity
- Fast, offline, privacy-safe — no API calls
- Benchmarked 2.8x faster with batch encoding

### Layer 2 — LLM-as-judge scoring (`llm_judge.py`)
- Semantic evaluation using structured prompts (G-Eval style)
- Supports faithfulness, relevancy, coherence metrics
- Robust score parsing — 3 fallback layers (JSON, markdown, regex)
- Catches logical contradictions that embeddings miss
- Swap MockLLM for any real LLM API in production

## Why two layers?
Layer 1 is fast and cheap — runs offline on every response.
Layer 2 is semantic — catches failures Layer 1 misses, like 
logical contradictions that are semantically similar but 
factually wrong (e.g. Sony vs Dolby Labs).

## Run
```bash
pip install sentence-transformers numpy
python eval_pipeline.py
```
