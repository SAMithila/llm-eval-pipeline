# LLM Evaluation Pipeline

Offline 3-metric evaluation pipeline for LLM outputs built for Apple Media Services use cases.
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

## Run
```bash
pip install sentence-transformers numpy
python eval_pipeline.py
```
