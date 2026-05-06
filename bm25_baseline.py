"""
models/bm25_baseline.py
-----------------------
Pure BM25 ranker used as the offline evaluation baseline.
Ranks documents by their full-document BM25 score against the query.
"""

import numpy as np
import pandas as pd
from features.bm25_features import BM25Scorer


class BM25Baseline:
    """
    Wrap a fitted BM25Scorer to produce ranked lists for evaluation.
    """

    def __init__(self, scorer: BM25Scorer | None = None):
        self.scorer = scorer or BM25Scorer()

    def fit(self, corpus: list[str]) -> "BM25Baseline":
        self.scorer.fit(corpus)
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Return BM25 score for each row (query, doc) in df."""
        queries = df["query"].tolist()
        docs    = (
            df["doc_title"].fillna("") + " " + df["doc_body"].fillna("")
        ).tolist()
        return self.scorer.batch_score(queries, docs)

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'bm25_score' column and return df sorted by
        (query_id, bm25_score DESC).
        """
        df = df.copy()
        df["bm25_score"] = self.predict_scores(df)
        return df.sort_values(
            ["query_id", "bm25_score"], ascending=[True, False]
        ).reset_index(drop=True)
