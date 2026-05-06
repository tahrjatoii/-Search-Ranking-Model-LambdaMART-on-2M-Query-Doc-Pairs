"""
models/lambdamart.py
---------------------
LambdaMART model for Learning-to-Rank.

Uses LightGBM's `lambdarank` objective which directly optimises NDCG.
The gradient/Hessian pairs are computed using the LambdaMART
formulation (pairwise losses weighted by NDCG delta).

Key hyper-parameters (tuned via grid search on val NDCG@10):
  n_estimators   500
  max_depth      7
  learning_rate  0.05
  num_leaves     127
  min_data_in_leaf 50
  lambdarank_truncation_level 10   → focuses on top-10 NDCG
"""

import numpy as np
import lightgbm as lgb
import joblib
from typing import Optional


class LambdaMARTRanker:
    """
    LambdaMART Learning-to-Rank model.

    Wraps LightGBM with the `lambdarank` objective and exposes a
    sklearn-style fit / predict interface.
    """

    DEFAULT_PARAMS = {
        "objective":                  "lambdarank",
        "metric":                     "ndcg",
        "ndcg_eval_at":               [1, 3, 5, 10],
        "lambdarank_truncation_level": 10,   # optimise NDCG@10
        "n_estimators":               500,
        "max_depth":                  7,
        "num_leaves":                 127,
        "learning_rate":              0.05,
        "min_data_in_leaf":           50,
        "feature_fraction":           0.8,
        "bagging_fraction":           0.8,
        "bagging_freq":               5,
        "reg_alpha":                  0.1,
        "reg_lambda":                 0.1,
        "verbosity":                  -1,
        "n_jobs":                     -1,
        "random_state":               42,
    }

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[lgb.LGBMRanker] = None
        self._feature_names: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val:   np.ndarray | None = None,
        y_val:   np.ndarray | None = None,
        groups_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        early_stopping_rounds: int = 50,
    ) -> "LambdaMARTRanker":

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        self.model = lgb.LGBMRanker(**self.params)

        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        ]

        eval_set = None
        eval_group = None
        if X_val is not None:
            eval_set  = [(X_val, y_val)]
            eval_group = [groups_val]

        self.model.fit(
            X_train, y_train,
            group=groups_train,
            eval_set=eval_set,
            eval_group=eval_group,
            feature_name=self._feature_names,
            callbacks=callbacks,
        )

        print(f"\n✅ LambdaMART trained. Best iteration: {self.model.best_iteration_}")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return relevance scores for each row (higher = more relevant)."""
        assert self.model is not None, "Call fit() first."
        return self.model.predict(X)

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self, importance_type: str = "gain") -> dict:
        assert self.model is not None
        imp = self.model.feature_importances_
        return dict(sorted(
            zip(self._feature_names, imp),
            key=lambda x: -x[1],
        ))

    def print_feature_importance(self, top_n: int = 20) -> None:
        fi = self.feature_importance("gain")
        print(f"\n{'Feature':<35} {'Importance':>12}")
        print("─" * 50)
        for name, score in list(fi.items())[:top_n]:
            print(f"  {name:<33} {score:>12.1f}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        print(f"LambdaMARTRanker saved → {path}")

    @classmethod
    def load(cls, path: str) -> "LambdaMARTRanker":
        model = joblib.load(path)
        print(f"LambdaMARTRanker loaded ← {path}")
        return model
