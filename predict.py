"""
predict.py
-----------
Production inference: re-rank a set of candidate documents for a query
using the trained LambdaMART model.

Usage:
    python predict.py \\
        --model_path artifacts/lambdamart.pkl \\
        --feature_store artifacts/feature_store.pkl \\
        --query "senior python engineer machine learning" \\
        --top_k 10
"""

import argparse
import json
import numpy as np
import pandas as pd

from models.lambdamart      import LambdaMARTRanker
from features.feature_store import FeatureStore


def build_candidate_df(query: str, candidates: list[dict], query_id: str = "q_live") -> pd.DataFrame:
    """
    Convert a list of candidate documents into the DataFrame schema
    expected by FeatureStore.transform().
    Each candidate dict should have: doc_title, doc_body, doc_skills,
    years_exp, num_bullets, company, degree, and all user signal fields.
    """
    rows = []
    for i, cand in enumerate(candidates):
        rows.append({
            "query_id":          query_id,
            "doc_id":            cand.get("doc_id", f"cand_{i}"),
            "query":             query,
            "relevance":         0,           # unknown at inference time
            "doc_title":         cand.get("doc_title", ""),
            "doc_body":          cand.get("doc_body", ""),
            "doc_skills":        cand.get("doc_skills", ""),
            "years_exp":         cand.get("years_exp", 0),
            "num_bullets":       cand.get("num_bullets", 0),
            "company":           cand.get("company", ""),
            "degree":            cand.get("degree", ""),
            # User signals (default to neutral values at inference)
            "ctr":               cand.get("ctr", 0.05),
            "dwell_time_sec":    cand.get("dwell_time_sec", 30.0),
            "skip_rate":         cand.get("skip_rate", 0.5),
            "conversion_rate":   cand.get("conversion_rate", 0.05),
            "bookmark_rate":     cand.get("bookmark_rate", 0.01),
            "impressions":       cand.get("impressions", 100),
            "clicks":            cand.get("clicks", 5),
            "avg_position":      cand.get("avg_position", 5.0),
        })
    return pd.DataFrame(rows)


def rerank(
    query: str,
    candidates: list[dict],
    model: LambdaMARTRanker,
    feature_store: FeatureStore,
    top_k: int = 10,
) -> list[dict]:
    """
    Re-rank candidates for a query.
    Returns top_k candidates sorted by LambdaMART score descending.
    """
    df = build_candidate_df(query, candidates)
    X, _, _ = feature_store.transform(df, verbose=False)
    scores = model.predict(X)

    results = []
    for i, (cand, score) in enumerate(zip(candidates, scores)):
        results.append({**cand, "_ltr_score": float(score), "_rank": 0})

    results.sort(key=lambda x: -x["_ltr_score"])
    for rank, r in enumerate(results[:top_k], 1):
        r["_rank"] = rank

    return results[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",      default="artifacts/lambdamart.pkl")
    parser.add_argument("--feature_store",   default="artifacts/feature_store.pkl")
    parser.add_argument("--query",           required=True)
    parser.add_argument("--candidates_file", default=None,
                        help="JSON file with list of candidate dicts")
    parser.add_argument("--top_k",           type=int, default=10)
    args = parser.parse_args()

    model = LambdaMARTRanker.load(args.model_path)
    fs    = FeatureStore.load(args.feature_store)

    if args.candidates_file:
        with open(args.candidates_file) as f:
            candidates = json.load(f)
    else:
        # Demo candidates
        candidates = [
            {
                "doc_id": "demo_001",
                "doc_title": "Senior Python Engineer Machine Learning",
                "doc_body": "Built pytorch pipeline processing 50M events/day. Led team of 5 engineers.",
                "doc_skills": "python|pytorch|kubernetes|spark",
                "years_exp": 7, "num_bullets": 5, "company": "Google",
            },
            {
                "doc_id": "demo_002",
                "doc_title": "Junior Java Developer",
                "doc_body": "Entry-level Java development. 1 year experience.",
                "doc_skills": "java|spring",
                "years_exp": 1, "num_bullets": 2, "company": "Startup",
            },
            {
                "doc_id": "demo_003",
                "doc_title": "ML Engineer NLP",
                "doc_body": "Implemented LLM serving system. Python, transformers, FastAPI.",
                "doc_skills": "python|llm|nlp|transformers",
                "years_exp": 4, "num_bullets": 4, "company": "Anthropic",
            },
        ]

    ranked = rerank(args.query, candidates, model, fs, top_k=args.top_k)

    print(f"\n🔍 Query: {args.query!r}")
    print(f"{'Rank':<6} {'Score':>8}  {'Title'}")
    print("─" * 60)
    for r in ranked:
        print(f"  {r['_rank']:<4} {r['_ltr_score']:>8.4f}  {r.get('doc_title', r.get('doc_id'))}")


if __name__ == "__main__":
    main()
