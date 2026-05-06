"""
features/user_signals.py
-------------------------
Extracts and normalises user engagement features from click/dwell logs.

Features:
  - ctr                  click-through rate (raw + log-odds transform)
  - dwell_time_sec       raw dwell time (seconds)
  - dwell_norm           dwell time normalised by query-level max
  - skip_rate            fraction of impressions with no click
  - conversion_rate      fraction of clicks leading to a conversion
  - bookmark_rate        fraction of clicks leading to a bookmark
  - engagement_score     composite signal (weighted combination)
  - position_bias_adj    position-de-biased CTR (propensity correction)
"""

import numpy as np
import pandas as pd


# ── Position propensity weights (estimated from randomised experiments) ──────
# P(click | position, not relevant)  — decreases with rank
POSITION_PROPENSITY = {
    1: 1.00, 2: 0.68, 3: 0.48, 4: 0.35, 5: 0.27,
    6: 0.21, 7: 0.17, 8: 0.14, 9: 0.12, 10: 0.10,
}


def _position_propensity(pos: float) -> float:
    p = max(1, min(10, round(pos)))
    return POSITION_PROPENSITY.get(p, 0.10)


def add_user_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 1. CTR log-odds (more linear for tree models) ────────────────────────
    eps = 1e-6
    df["ctr_logodds"] = np.log((df["ctr"] + eps) / (1 - df["ctr"] + eps))

    # ── 2. Dwell time: log-scale + query-level normalisation ─────────────────
    df["log_dwell"] = np.log1p(df["dwell_time_sec"])
    query_max_dwell = df.groupby("query_id")["dwell_time_sec"].transform("max").clip(lower=1)
    df["dwell_norm"] = df["dwell_time_sec"] / query_max_dwell

    # ── 3. Position-bias-adjusted CTR  ───────────────────────────────────────
    propensity = df["avg_position"].apply(_position_propensity)
    df["ctr_position_adj"] = (df["ctr"] / propensity).clip(upper=1.0)

    # ── 4. Engagement composite score ────────────────────────────────────────
    # Weights tuned on human eval judgement correlation
    df["engagement_score"] = (
        0.35 * df["ctr_position_adj"] +
        0.30 * df["dwell_norm"] +
        0.20 * (1 - df["skip_rate"]) +
        0.10 * df["conversion_rate"] +
        0.05 * df["bookmark_rate"]
    )

    # ── 5. Click volume signal (log) ─────────────────────────────────────────
    df["log_clicks"]      = np.log1p(df["clicks"])
    df["log_impressions"] = np.log1p(df["impressions"])

    return df


USER_SIGNAL_FEATURES = [
    "ctr",
    "ctr_logodds",
    "dwell_time_sec",
    "log_dwell",
    "dwell_norm",
    "skip_rate",
    "conversion_rate",
    "bookmark_rate",
    "ctr_position_adj",
    "engagement_score",
    "log_clicks",
    "log_impressions",
    "avg_position",
]
