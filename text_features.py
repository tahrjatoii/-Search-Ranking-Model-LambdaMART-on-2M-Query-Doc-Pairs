"""
features/text_features.py
--------------------------
Text-level relevance features beyond BM25:
  - tfidf_cosine      TF-IDF cosine similarity (query vs doc)
  - exact_match_frac  fraction of query terms exactly in doc
  - query_coverage    fraction of query bigrams covered by doc
  - idf_weighted_overlap  sum of IDF weights for matching terms
  - title_exact_match strict match of query terms in title only
"""

import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bigrams(tokens: List[str]) -> List[tuple]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


class TextFeatureExtractor:

    def __init__(self):
        self._idf: Dict[str, float] = {}
        self._corpus_size: int = 0

    def fit(self, corpus: List[str]) -> "TextFeatureExtractor":
        """Build IDF weights from corpus."""
        self._corpus_size = len(corpus)
        doc_freq: Counter = Counter()
        for text in corpus:
            terms = set(_tokenize(text))
            doc_freq.update(terms)
        for term, df in doc_freq.items():
            self._idf[term] = math.log(
                (self._corpus_size + 1) / (df + 1)
            ) + 1.0
        return self

    def _tfidf_vector(self, text: str) -> Dict[str, float]:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        length = len(tokens)
        vec = {}
        for t, c in tf.items():
            vec[t] = (c / max(length, 1)) * self._idf.get(t, 1.0)
        return vec

    def cosine_sim(self, query: str, doc: str) -> float:
        qv = self._tfidf_vector(query)
        dv = self._tfidf_vector(doc)
        common = set(qv) & set(dv)
        if not common:
            return 0.0
        dot = sum(qv[t] * dv[t] for t in common)
        q_norm = math.sqrt(sum(v ** 2 for v in qv.values()))
        d_norm = math.sqrt(sum(v ** 2 for v in dv.values()))
        return dot / (q_norm * d_norm + 1e-9)

    def exact_match_frac(self, query: str, doc: str) -> float:
        q_terms = set(_tokenize(query))
        d_terms = set(_tokenize(doc))
        if not q_terms:
            return 0.0
        return len(q_terms & d_terms) / len(q_terms)

    def query_coverage(self, query: str, doc: str) -> float:
        """Fraction of query bigrams present in the document."""
        q_bigrams = _bigrams(_tokenize(query))
        if not q_bigrams:
            return self.exact_match_frac(query, doc)
        d_text = " ".join(_tokenize(doc))
        covered = sum(
            1 for (a, b) in q_bigrams if f"{a} {b}" in d_text
        )
        return covered / len(q_bigrams)

    def idf_weighted_overlap(self, query: str, doc: str) -> float:
        q_terms = set(_tokenize(query))
        d_terms = set(_tokenize(doc))
        return sum(self._idf.get(t, 1.0) for t in q_terms & d_terms)

    def title_exact_match(self, query: str, title: str) -> float:
        return self.exact_match_frac(query, title)

    def extract_all(self, query: str, doc_body: str, doc_title: str) -> dict:
        full_doc = doc_title + " " + doc_body
        return {
            "tfidf_cosine":          self.cosine_sim(query, full_doc),
            "exact_match_frac":      self.exact_match_frac(query, full_doc),
            "query_coverage":        self.query_coverage(query, full_doc),
            "idf_weighted_overlap":  self.idf_weighted_overlap(query, full_doc),
            "title_exact_match":     self.title_exact_match(query, doc_title),
        }

    def batch_extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised extraction; returns df with new feature columns."""
        results = []
        for _, row in df.iterrows():
            feats = self.extract_all(
                row["query"], row["doc_body"], row["doc_title"]
            )
            results.append(feats)
        feat_df = pd.DataFrame(results, index=df.index)
        return pd.concat([df, feat_df], axis=1)


TEXT_FEATURES = [
    "tfidf_cosine",
    "exact_match_frac",
    "query_coverage",
    "idf_weighted_overlap",
    "title_exact_match",
]
