"""
features/bm25_features.py
--------------------------
Computes BM25 scores as features for query-document pairs.

We build a lightweight per-field BM25 scorer without requiring
an external search engine.
"""

import math
import numpy as np
import pandas as pd
from collections import Counter
from typing import List


# ── Okapi BM25 parameters ────────────────────────────────────────────────────

K1 = 1.5   # term frequency saturation
B  = 0.75  # document length normalization


def _tokenize(text: str) -> List[str]:
    """Lowercase split; punctuation stripped."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Scorer:
    """
    Corpus-level BM25 scorer.
    Call fit() once on the corpus, then score() per (query, doc) pair.
    """

    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b  = b
        self._idf: dict = {}
        self._avg_len: float = 1.0
        self._n_docs: int = 0

    def fit(self, corpus: List[str]) -> "BM25Scorer":
        """Compute IDF weights and average document length from corpus."""
        self._n_docs = len(corpus)
        lengths = []
        doc_freq: Counter = Counter()

        for text in corpus:
            tokens = set(_tokenize(text))
            doc_freq.update(tokens)
            lengths.append(len(_tokenize(text)))

        self._avg_len = float(np.mean(lengths)) if lengths else 1.0

        for term, df in doc_freq.items():
            # Robertson / Sparck Jones IDF
            self._idf[term] = math.log(
                (self._n_docs - df + 0.5) / (df + 0.5) + 1.0
            )
        return self

    def score(self, query: str, doc: str) -> float:
        """BM25 score for a single (query, doc) pair."""
        if not self._idf:
            return 0.0
        q_tokens = _tokenize(query)
        d_tokens = _tokenize(doc)
        d_len    = len(d_tokens)
        tf_map   = Counter(d_tokens)

        score = 0.0
        for term in set(q_tokens):
            idf = self._idf.get(term, 0.0)
            tf  = tf_map.get(term, 0)
            numerator   = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * d_len / self._avg_len)
            score += idf * numerator / (denominator + 1e-9)
        return score

    def batch_score(self, queries: List[str], docs: List[str]) -> np.ndarray:
        return np.array([self.score(q, d) for q, d in zip(queries, docs)])


def add_bm25_features(
    df: pd.DataFrame,
    title_scorer: BM25Scorer,
    body_scorer: BM25Scorer,
    full_scorer: BM25Scorer,
) -> pd.DataFrame:
    """
    Add three BM25 score columns to df:
      bm25_title  — score against doc title only
      bm25_body   — score against doc body
      bm25_full   — score against title + body concatenated
    """
    df = df.copy()
    queries = df["query"].tolist()
    titles  = df["doc_title"].fillna("").tolist()
    bodies  = df["doc_body"].fillna("").tolist()
    fulls   = (df["doc_title"].fillna("") + " " + df["doc_body"].fillna("")).tolist()

    df["bm25_title"] = title_scorer.batch_score(queries, titles)
    df["bm25_body"]  = body_scorer.batch_score(queries, bodies)
    df["bm25_full"]  = full_scorer.batch_score(queries, fulls)
    return df


def fit_bm25_scorers(
    train_df: pd.DataFrame,
) -> tuple[BM25Scorer, BM25Scorer, BM25Scorer]:
    """Fit three BM25 scorers on the training corpus."""
    print("Fitting BM25 scorers on training corpus...")

    title_corpus = train_df["doc_title"].fillna("").tolist()
    body_corpus  = train_df["doc_body"].fillna("").tolist()
    full_corpus  = (
        train_df["doc_title"].fillna("") + " " + train_df["doc_body"].fillna("")
    ).tolist()

    title_scorer = BM25Scorer().fit(title_corpus)
    body_scorer  = BM25Scorer().fit(body_corpus)
    full_scorer  = BM25Scorer().fit(full_corpus)

    print(f"  Title vocab: {len(title_scorer._idf):,} terms")
    print(f"  Body  vocab: {len(body_scorer._idf):,}  terms")
    return title_scorer, body_scorer, full_scorer
