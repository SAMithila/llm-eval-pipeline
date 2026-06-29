import numpy as np
from sentence_transformers import SentenceTransformer
import csv
import os
from datetime import datetime

# 1. Load a free local embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')   # downloads once, 

# 2. Your cosine similarity function
def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b + 1e-9)

# 3. Faithfulness scorer
def faithfulness_score(response: str, context: str, threshold=0.7) -> dict:
    """Does the response stay faithful to the retrieved context?"""
    resp_emb = model.encode(response)
    ctx_emb = model.encode(context)
    score = cosine_similarity(resp_emb, ctx_emb)
    return {
        "score": round(float(score), 3),
        "pass": score >= threshold,
        "label": "✅ faithful" if score >= threshold else " ❌ hallucination_risk"
    }

# Relevancy scorer
def score_relevancy(query: str, response: str, threshold = 0.6) -> dict:
    q_emb = model.encode(query)
    resp_emb = model.encode(response)
    score = cosine_similarity(q_emb, resp_emb)
    return{
        "score": round(float(score), 3),
        "pass": score >= threshold,
        "label": "✅ relevant" if score >= threshold else " ❌ off_topic"
    }

# Coherence scorer
def score_coherence(response: str) -> dict:
    """
    Measures logical flow between consecutive sentences.
    Uses length-aware threshold — short responses get lower bar
    since topic switches between 2 sentences are normal.
    """
    sentences = [s.strip() for s in response.split('.')
                 if s.strip()]

    # Single sentence — trivially coherent
    if len(sentences) < 3:
        return {
            "score": 1.0,
            "pass":  True,
            "label": "✅ coherent",
            "note":  "single sentence"
        }

    # Embed all sentences in one batch call
    embeddings = model.encode(sentences)

    # Consecutive sentence similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)

    score = float(np.mean(similarities))

    # Length-aware threshold:
    # 2 sentences → low bar (0.3) — topic switch is normal
    # 3+ sentences → higher bar (0.6) — sustained flow expected
    threshold = 0.3 if len(sentences) == 2 else 0.6

    return {
        "score": round(score, 3),
        "pass":  score >= threshold,
        "label": "✅ coherent" if score >= threshold else "❌ incoherent"
    }

# Ten test cases
test_cases = [
    {
        "query":    "What is the refund policy for Apple Music?",
        "response": "Apple Music offers a 30-day free trial. Refunds are not provided for partial months.",
        "context":  "Apple Music subscriptions offer a 30-day trial. No refunds for partial periods."
    },
    {
        "query":    "How do I cancel my Apple TV+ subscription?",
        "response": "You can cancel Apple TV+ anytime via Settings > Apple ID > Subscriptions.",
        "context":  "Apple TV+ can be cancelled through your device Settings under Apple ID and Subscriptions."
    },
    {
        "query":    "What audio quality does Apple Music support?",
        "response": "Apple Music supports lossless audio up to 24-bit/192kHz and Dolby Atmos.",
        "context":  "Apple Music offers Lossless Audio (up to 24-bit/48kHz) and Hi-Res Lossless (up to 24-bit/192kHz) plus Spatial Audio with Dolby Atmos."
    },
    {
        "query":    "Can I share Apple Music with family?",
        "response": "Yes, Apple Music Family plan supports up to 6 members.",
        "context":  "Apple One and Apple Music Family plans allow sharing with up to 6 family members via Family Sharing."
    },
    {
        "query":    "Does Apple Music work offline?",
        "response": "Yes, you can download songs for offline listening in the Apple Music app.",
        "context":  "Apple Music allows users to download songs, albums, and playlists for offline playback."
    },
    {
        "query":    "What is Apple Podcasts?",
        "response": "Apple Podcasts is a free app for discovering and listening to podcasts.",
        "context":  "Apple Podcasts is Apple's platform for podcast discovery, subscription, and playback, available free on all Apple devices."
    },
    {
        "query":    "How much does Apple Music cost?",
        "response": "Apple Music costs $10.99 per month for individuals in the US.",
        "context":  "Apple Music individual plan is priced at $10.99/month in the United States."
    },
    {
        "query":    "Can I try Apple TV+ for free?",
        "response": "Apple TV+ offers a 7-day free trial for new subscribers.",
        "context":  "New Apple TV+ subscribers get a 7-day free trial before being charged."
    },
    {
        "query":    "What is Spatial Audio in Apple Music?",
        # ← intentional hallucination — let's see if your scorer catches it
        "response": "Spatial Audio uses 8D technology developed by Sony to create surround sound.",
        "context":  "Spatial Audio with Dolby Atmos creates a three-dimensional listening experience, developed with Dolby Labs and available in Apple Music."
    },
    {
        "query":    "How do I get Apple One?",
        "response": "Apple One bundles Apple Music, TV+, Arcade, and iCloud+ in one subscription.",
        "context":  "Apple One is a bundled subscription service combining Apple Music, Apple TV+, Apple Arcade, and iCloud+ at a discounted price."
    },
]

# Part B: CSV Logging
def save_to_csv (results: list, filename= "eval_results.csv"):
    """Save evalution results to CSV with timestamp)"""

    file_exits = os.path.exists(filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "test_id", "query", "faithfulness_score", "faithfulness_pass", "relevancy_score", "relevancy_pass", "coherence_score", "coherence_pass", "overall_pass"
        ])

        # Write header only if file is new
        if not file_exits:
            writer.writeheader()

        for r in results:
            writer.writerow({
                "timestamp":          timestamp,
                "test_id":            r["test_id"],
                "query":              r["query"][:60],
                "faithfulness_score": r["faithfulness"]["score"], 
                "faithfulness_pass":  r["faithfulness"]["pass"],
                "relevancy_score":    r["relevancy"]["score"], 
                "relevancy_pass":     r["relevancy"]["pass"],
                "coherence_score":    r["coherence"]["score"],
                "coherence_pass":  r["coherence"]["pass"],
                "overall_pass":       r["overall_pass"]
            })
    print(f"\n Results saved -> {filename}")


# Part B: Summary Dashboard
def print_dashboard(results:list):
    """Print a human readable summary dashboard of the evaluation results"""
    total    = len(results)
    f_scores = [r["faithfulness"]["score"] for r in results]
    r_scores = [r["relevancy"]["score"] for r in results]
    f_passed = sum(1 for r in results if r["faithfulness"]["pass"])
    r_passed = sum(1 for r in results if r["relevancy"]["pass"])
    c_scores = [r["coherence"]["score"] for r in results]
    c_passed = sum(1 for r in results if r["coherence"]["pass"])
    overall  = sum(1 for r in results if r["overall_pass"])
    failed   = [r for r in results if not r["overall_pass"]]

    print("\n" + "="*60)
    print(" Evalution Dashboard")
    print("\n" + "="*60)
    print(f" Run timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Total test cases: {total}")
    print()

    print(f" Faithfulness")
    print(f" Pass rate: {f_passed}/{total} ({f_passed/total*100:.0f}%)")
    print(f" Avg score: {np.mean(f_scores):.3f}")
    print(f" Min score: {np.min(r_scores):.3f} ← worst case")
    print()

    print(f" Relevancy")
    print(f" Pass rate: {r_passed}/{total} ({r_passed/total*100:.0f}%)")
    print(f" Avg score: {np.mean(r_scores):.3f}")
    print(f" Min score: {np.min(r_scores):.3f} ← worst case")
    print()

    print(f"  Coherence")
    print(f"    Pass rate       : {c_passed}/{total}  ({c_passed/total*100:.0f}%)")
    print(f"    Avg score       : {np.mean(c_scores):.3f}")
    print(f"    Min score       : {np.min(c_scores):.3f}  ← worst case")
    print()

    print(f" Overall pass rate: {overall}/{total} ({overall/total*100:.0f}%)")
    print()

    if failed:
        print(f" Failed cases ({len(failed)}):")
        for r in failed:
            print(f" Test{r['test_id']:02d}: {r['query'][:45]}...")
            if not r["faithfulness"]["pass"]:
                print(f"  Faithfulness: {r['faithfulness']['score']} ← below threshold")
            if not r["relevancy"]["pass"]:
                print(f"  Relevancy:    {r['relevancy']['score']} ← below threshold")
        
    print("\n" + "="*60)
    
"""
# Run evaluation + print report (First test run)
print("\n" + "="*60)
print("  APPLE MEDIA SERVICES — LLM EVALUATION REPORT")
print("\n" + "="*60)

passed = 0
failed = 0

for i, tc in enumerate(test_cases):
    faith = faithfulness_score(tc["response"], tc["context"])
    relev = score_relevancy(tc["query"], tc["response"])
    overall = "PASS" if faith["pass"] and relev["pass"] else "FAIL"
    if faith["pass"] and relev["pass"]:
        passed += 1
    else:
        failed += 1
    print(f"\nTest {i+1}: {tc['query'][:50]}...")
    print(f" Faithfulness : {faith['score']} {faith['label']}")
    print(f" Relevancy   : {relev['score']} {relev['label']}")
    print(f" Overall : {overall}")

print("\n" + "="*60)
print(f" SUMMARY: {passed}/10 passed | {failed}/10 failed")
print(f" Hallucination caught in test 9: "
      f" {'Yes✅' if not faithfulness_score(test_cases[8]['response'], test_cases[8]['context'])['pass'] else 'No ❌'}")
print("\n" + "="*60)
"""

# Collects results for dashboard + CSV logging

print("\n" + "="*60)
print("  APPLE MEDIA SERVICES — LLM EVALUATION REPORT")
print("\n" + "="*60)

results = []

for i, tc in enumerate(test_cases):
    faith = faithfulness_score(tc["response"], tc["context"])
    relev = score_relevancy(tc["query"], tc["response"])
    coher   = score_coherence(tc["response"])             

    overall_pass = faith["pass"] and relev["pass"] and coher["pass"] 

    results.append({
        "test_id":      i + 1,
        "query":        tc["query"],
        "faithfulness": faith,
        "relevancy":    relev,
        "coherence":    coher,                            
        "overall_pass": overall_pass
    })

    print(f"\nTest {i+1:02d}: {tc['query'][:50]}...")
    print(f"  Faithfulness : {faith['score']}  {faith['label']}")
    print(f"  Relevancy    : {relev['score']}  {relev['label']}")
    print(f"  Coherence    : {coher['score']}  {coher['label']}")  
    print(f"  Overall      : {'✅ PASS' if overall_pass else '❌ FAIL'}")


# Run dashboard + save CSV
print_dashboard(results)
save_to_csv(results)
