"""
train.py
---------
End-to-end training pipeline:

  1. Load 2M+ query-doc pairs from parquet chunks
  2. Split into train / val / test at query level
  3. Fit FeatureStore (BM25, TF-IDF, user signals)
  4. Train BM25 baseline
  5. Train LambdaMART with early stopping on val NDCG@10
  6. Evaluate on held-out test set and print comparison table
  7. Save model artifacts

Usage:
    python train.py --data_dir data/raw/ --model_dir artifacts/ --eval
    python train.py --data_dir data/raw/ --model_dir artifacts/ --max_chunks 5
"""

import os
import argparse
import time
import numpy as np
import pandas as pd

from data.loader          import load_raw_chunks, query_level_split
from features.feature_store import FeatureStore, ALL_FEATURES
from models.bm25_baseline  import BM25Baseline
from models.lambdamart     import LambdaMARTRanker
from evaluation.evaluator  import run_full_evaluation


def parse_args():
    p = argparse.ArgumentParser(description="Train LambdaMART search ranker")
    p.add_argument("--data_dir",    default="data/raw/",  help="Directory with chunk parquets")
    p.add_argument("--model_dir",   default="artifacts/", help="Where to save model artifacts")
    p.add_argument("--max_chunks",  type=int, default=None, help="Limit number of chunks loaded")
    p.add_argument("--val_frac",    type=float, default=0.10)
    p.add_argument("--test_frac",   type=float, default=0.10)
    p.add_argument("--n_estimators",type=int, default=500)
    p.add_argument("--max_depth",   type=int, default=7)
    p.add_argument("--lr",          type=float, default=0.05)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--eval",        action="store_true", help="Run evaluation on test set")
    p.add_argument("--no_save",     action="store_true", help="Skip saving artifacts")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 1 — Loading data")
    print("─" * 60)
    df = load_raw_chunks(args.data_dir, max_chunks=args.max_chunks)
    print(f"Total pairs: {len(df):,}  |  Queries: {df['query_id'].nunique():,}")
    print(f"Relevance distribution:\n{df['relevance'].value_counts().sort_index()}\n")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    print("─" * 60)
    print("STEP 2 — Query-level train / val / test split")
    print("─" * 60)
    train_df, val_df, test_df = query_level_split(
        df,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # ── 3. Feature Store ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 3 — Fitting FeatureStore")
    print("─" * 60)
    fs = FeatureStore()
    fs.fit(train_df)

    print("\nTransforming train...")
    X_train, y_train, g_train = fs.transform(train_df)

    print("Transforming val...")
    X_val,   y_val,   g_val   = fs.transform(val_df)

    print(f"\nFeature matrix: {X_train.shape}  |  Features: {len(ALL_FEATURES)}")
    print(f"Feature names: {ALL_FEATURES[:5]} ... [{len(ALL_FEATURES)} total]")

    # ── 4. BM25 baseline ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 4 — Fitting BM25 Baseline")
    print("─" * 60)
    bm25_corpus = (
        train_df["doc_title"].fillna("") + " " +
        train_df["doc_body"].fillna("")
    ).tolist()
    bm25 = BM25Baseline()
    bm25.fit(bm25_corpus)

    # ── 5. LambdaMART ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 5 — Training LambdaMART")
    print("─" * 60)
    ltr = LambdaMARTRanker(params={
        "n_estimators": args.n_estimators,
        "max_depth":    args.max_depth,
        "learning_rate":args.lr,
    })
    ltr.fit(
        X_train, y_train, g_train,
        X_val=X_val, y_val=y_val, groups_val=g_val,
        feature_names=ALL_FEATURES,
    )
    ltr.print_feature_importance(top_n=15)

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    if args.eval:
        print("\n" + "─" * 60)
        print("STEP 6 — Evaluation on held-out test set")
        print("─" * 60)
        bm25_metrics, ltr_metrics = run_full_evaluation(
            bm25_ranker=bm25,
            ltr_ranker=ltr,
            feature_store=fs,
            test_df=test_df,
            k=10,
        )

        # Summarise improvement
        ndcg_bm25 = bm25_metrics["NDCG@10"]
        ndcg_ltr  = ltr_metrics["NDCG@10"]
        pct = (ndcg_ltr - ndcg_bm25) / ndcg_bm25 * 100
        print(f"🎯 NDCG@10 improvement: {ndcg_bm25:.4f} → {ndcg_ltr:.4f}  (+{pct:.1f}%)")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    if not args.no_save:
        print("\n" + "─" * 60)
        print("STEP 7 — Saving artifacts")
        print("─" * 60)
        fs.save(os.path.join(args.model_dir, "feature_store.pkl"))
        ltr.save(os.path.join(args.model_dir, "lambdamart.pkl"))

    elapsed = time.time() - t0
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
