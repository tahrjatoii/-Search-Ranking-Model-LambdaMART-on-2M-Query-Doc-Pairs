"""
evaluation/metrics.py
----------------------
Ranking evaluation metrics implemented from scratch:

  ndcg_at_k      Normalised Discounted Cumulative Gain @ k
  map_at_k       Mean Average Precision @ k
  mrr            Mean Reciprocal Rank
  precision_at_k Precision @ k
  recall_at_k    Recall @ k

All functions accept query-level lists of relevance scores (in ranked order)
and compute per-query then macro-average.
"""

import math
import numpy as np
from typing import List


# ── Per-query metrics ─────────────────────────────────────────────────────────

def dcg(rels: List[int], k: int) -> float:
    """Discounted Cumulative Gain @ k using log2 discount."""
    return sum(
        (2 ** r - 1) / math.log2(i + 2)
        for i, r in enumerate(rels[:k])
    )


def ndcg_query(rels: List[int], k: int) -> float:
    """NDCG@k for a single query."""
    ideal = sorted(rels, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(rels, k) / ideal_dcg


def ap_query(rels: List[int], k: int, relevant_threshold: int = 1) -> float:
    """Average Precision @ k for a single query."""
    hits, sum_prec = 0, 0.0
    for i, r in enumerate(rels[:k]):
        if r >= relevant_threshold:
            hits += 1
            sum_prec += hits / (i + 1)
    n_relevant = sum(1 for r in rels if r >= relevant_threshold)
    if n_relevant == 0:
        return 0.0
    return sum_prec / min(n_relevant, k)


def rr_query(rels: List[int], relevant_threshold: int = 1) -> float:
    """Reciprocal Rank for a single query."""
    for i, r in enumerate(rels):
        if r >= relevant_threshold:
            return 1.0 / (i + 1)
    return 0.0


def precision_query(rels: List[int], k: int, relevant_threshold: int = 1) -> float:
    return sum(1 for r in rels[:k] if r >= relevant_threshold) / k


def recall_query(rels: List[int], k: int, relevant_threshold: int = 1) -> float:
    n_rel = sum(1 for r in rels if r >= relevant_threshold)
    if n_rel == 0:
        return 0.0
    return sum(1 for r in rels[:k] if r >= relevant_threshold) / n_rel


# ── Corpus-level metrics ──────────────────────────────────────────────────────

def compute_metrics(
    ranked_rels: List[List[int]],
    k: int = 10,
) -> dict:
    """
    Args:
        ranked_rels  list of per-query relevance lists, already in ranked order
        k            cutoff for NDCG, MAP, Precision, Recall

    Returns:
        dict with mean NDCG@k, MAP@k, MRR, P@k, R@k and std for key metrics
    """
    ndcgs, aps, rrs, ps, rs = [], [], [], [], []

    for rels in ranked_rels:
        ndcgs.append(ndcg_query(rels, k))
        aps.append(ap_query(rels, k))
        rrs.append(rr_query(rels))
        ps.append(precision_query(rels, k))
        rs.append(recall_query(rels, k))

    return {
        f"NDCG@{k}":      float(np.mean(ndcgs)),
        f"NDCG@{k}_std":  float(np.std(ndcgs)),
        f"MAP@{k}":       float(np.mean(aps)),
        f"MRR":           float(np.mean(rrs)),
        f"P@{k}":         float(np.mean(ps)),
        f"R@{k}":         float(np.mean(rs)),
        "n_queries":      len(ranked_rels),
    }


def scores_to_ranked_rels(
    scores: np.ndarray,
    labels: np.ndarray,
    query_ids: np.ndarray,
) -> List[List[int]]:
    """
    Given parallel arrays of scores, labels, and query_ids,
    return per-query lists of labels sorted by descending score.
    """
    import pandas as pd
    df = pd.DataFrame({"qid": query_ids, "score": scores, "label": labels})
    ranked = []
    for _, grp in df.groupby("qid"):
        sorted_labels = grp.sort_values("score", ascending=False)["label"].tolist()
        ranked.append(sorted_labels)
    return ranked
