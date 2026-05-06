"""
features/feature_store.py
--------------------------
Orchestrates all feature groups into a single feature matrix
ready for LambdaMART training.

Feature groups:
  G1 — BM25 scores          (3 features)
  G2 — Text match features  (5 features)
  G3 — User signals         (13 features)
  G4 — Document quality     (4 features)
  ─────────────────────────────────────────
  Total                     25 features
"""

import numpy as np
import pandas as pd
import joblib
from typing import Optional

from features.bm25_features import (
    BM25Scorer, fit_bm25_scorers, add_bm25_features,
)
from features.user_signals import (
    add_user_signal_features, USER_SIGNAL_FEATURES,
)
from features.text_features import (
    TextFeatureExtractor, TEXT_FEATURES,
)

# ── Document quality features ────────────────────────────────────────────────

def add_doc_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Structural / quality signals derived from document metadata."""
    df = df.copy()

    # Body length normalised (log)
    body_len = df["doc_body"].fillna("").str.split().str.len()
    df["log_doc_len"] = np.log1p(body_len)

    # Experience signals
    df["years_exp_norm"] = (df["years_exp"] / 15.0).clip(0, 1)

    # Bullet density (bullets per 100 words)
    df["bullet_density"] = (
        df["num_bullets"] / body_len.clip(lower=1) * 100
    ).clip(0, 10)

    # Skill count (proxy for breadth)
    df["skill_count"] = df["doc_skills"].fillna("").str.split("|").str.len()

    return df


DOC_QUALITY_FEATURES = [
    "log_doc_len",
    "years_exp_norm",
    "bullet_density",
    "skill_count",
]

# ── All feature names ────────────────────────────────────────────────────────

BM25_FEATURES = ["bm25_title", "bm25_body", "bm25_full"]

ALL_FEATURES = (
    BM25_FEATURES +
    TEXT_FEATURES +
    USER_SIGNAL_FEATURES +
    DOC_QUALITY_FEATURES
)


class FeatureStore:
    """
    Fits on training data and transforms any split into
    a (X, y, groups) tuple ready for LightGBM lambdarank.
    """

    def __init__(self):
        self.title_bm25:   Optional[BM25Scorer] = None
        self.body_bm25:    Optional[BM25Scorer] = None
        self.full_bm25:    Optional[BM25Scorer] = None
        self.text_extractor: Optional[TextFeatureExtractor] = None
        self._fitted = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> "FeatureStore":
        print("Fitting FeatureStore on training data...")
        corpus = (
            train_df["doc_title"].fillna("") + " " +
            train_df["doc_body"].fillna("")
        ).tolist()

        self.title_bm25, self.body_bm25, self.full_bm25 = fit_bm25_scorers(train_df)

        print("Fitting TF-IDF text extractor...")
        self.text_extractor = TextFeatureExtractor().fit(corpus)

        self._fitted = True
        return self

    # ── Transform ────────────────────────────────────────────────────────────

    def transform(
        self,
        df: pd.DataFrame,
        batch_size: int = 50_000,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            X       (n, n_features)  float32 feature matrix
            y       (n,)             int32 relevance labels
            groups  (n_queries,)     int array — docs per query (for lambdarank)
        """
        assert self._fitted, "Call fit() before transform()"

        if verbose:
            print(f"Transforming {len(df):,} pairs into features...")

        chunks = []
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start:start + batch_size].copy()

            # G1 — BM25
            chunk = add_bm25_features(
                chunk, self.title_bm25, self.body_bm25, self.full_bm25
            )
            # G2 — Text match  (slow; batched for memory)
            chunk = self.text_extractor.batch_extract(chunk)

            # G3 — User signals
            chunk = add_user_signal_features(chunk)

            # G4 — Doc quality
            chunk = add_doc_quality_features(chunk)

            chunks.append(chunk)
            if verbose:
                done = min(start + batch_size, len(df))
                print(f"  {done:>10,} / {len(df):,}", end="\r")

        if verbose:
            print()

        full = pd.concat(chunks, ignore_index=True)

        # Fill any NaNs introduced during feature computation
        full[ALL_FEATURES] = full[ALL_FEATURES].fillna(0.0)

        X = full[ALL_FEATURES].values.astype(np.float32)
        y = full["relevance"].values.astype(np.int32)

        # Groups = number of docs per query (must be sorted by query_id)
        groups = (
            full.groupby("query_id", sort=False)["relevance"]
            .count()
            .values.astype(np.int32)
        )
        return X, y, groups

    # ── Persist ──────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        print(f"FeatureStore saved → {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureStore":
        fs = joblib.load(path)
        print(f"FeatureStore loaded ← {path}")
        return fs
