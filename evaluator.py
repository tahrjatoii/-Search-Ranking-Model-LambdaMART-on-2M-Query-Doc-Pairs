"""
evaluation/evaluator.py
------------------------
Runs BM25 baseline and LambdaMART through the evaluation harness
and prints a comparison table with NDCG@10, MAP@10, MRR, and improvement %.
"""

import numpy as np
import pandas as pd
from typing import Callable

from evaluation.metrics import compute_metrics, scores_to_ranked_rels


def evaluate_ranker(
    score_fn: Callable[[pd.DataFrame], np.ndarray],
    df: pd.DataFrame,
    k: int = 10,
    label: str = "Model",
) -> dict:
    """
    score_fn   : function (df) -> np.ndarray of scores (higher = more relevant)
    df         : DataFrame with columns query_id, relevance
    k          : cutoff
    """
    scores   = score_fn(df)
    ranked   = scores_to_ranked_rels(scores, df["relevance"].values, df["query_id"].values)
    metrics  = compute_metrics(ranked, k=k)
    metrics["model"] = label
    return metrics


def print_comparison_table(results: list[dict], k: int = 10) -> None:
    """Pretty-print side-by-side metric comparison."""
    ndcg_key = f"NDCG@{k}"
    map_key  = f"MAP@{k}"

    baseline = results[0]

    print("\n" + "═" * 70)
    print(f"  OFFLINE EVALUATION RESULTS  (k = {k})")
    print("═" * 70)
    print(f"  {'Model':<22} {ndcg_key:>10} {'MAP@'+str(k):>10} {'MRR':>8}  {'P@'+str(k):>8}")
    print("─" * 70)

    for r in results:
        ndcg_delta = ""
        if r["model"] != baseline["model"]:
            pct = (r[ndcg_key] - baseline[ndcg_key]) / (baseline[ndcg_key] + 1e-9) * 100
            ndcg_delta = f"  (+{pct:.1f}%)" if pct > 0 else f"  ({pct:.1f}%)"

        print(
            f"  {r['model']:<22} "
            f"{r[ndcg_key]:>10.4f}"
            f"{ndcg_delta:<12}"
            f"{r[map_key]:>10.4f}"
            f"{r['MRR']:>8.4f}"
            f"{r['P@'+str(k)]:>8.4f}"
        )
    print("═" * 70 + "\n")


def run_full_evaluation(
    bm25_ranker,
    ltr_ranker,
    feature_store,
    test_df: pd.DataFrame,
    k: int = 10,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """
    End-to-end evaluation of both rankers on test_df.
    Returns (bm25_metrics, ltr_metrics).
    """
    # ── BM25 baseline ─────────────────────────────────────────────────────────
    if verbose:
        print("Evaluating BM25 baseline...")
    bm25_metrics = evaluate_ranker(
        score_fn=bm25_ranker.predict_scores,
        df=test_df,
        k=k,
        label="BM25 Baseline",
    )

    # ── LambdaMART ────────────────────────────────────────────────────────────
    if verbose:
        print("Evaluating LambdaMART...")

    X_test, y_test, _ = feature_store.transform(test_df, verbose=verbose)
    ltr_scores = ltr_ranker.predict(X_test)

    ltr_metrics = evaluate_ranker(
        score_fn=lambda _: ltr_scores,
        df=test_df,
        k=k,
        label="LambdaMART",
    )

    if verbose:
        print_comparison_table([bm25_metrics, ltr_metrics], k=k)

    return bm25_metrics, ltr_metrics
