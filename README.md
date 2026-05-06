# 🔍 Search Ranking Model — LambdaMART on 2M+ Query-Doc Pairs

A production-grade **Learning to Rank (LTR)** pipeline that replaces a BM25 baseline with a LambdaMART model trained on 2M+ query-document pairs, achieving **+10% NDCG@10 improvement** in offline evaluation.

---

## 📁 Project Structure

```
search_ranking/
├── data/
│   ├── generator.py          # Synthetic 2M+ query-doc pair generation
│   └── loader.py             # Dataset loading & train/val/test splits
├── features/
│   ├── bm25_features.py      # BM25 scoring features
│   ├── user_signals.py       # Click, dwell-time, CTR, position bias features
│   ├── text_features.py      # TF-IDF cosine, exact match, query coverage
│   └── feature_store.py      # Feature assembly & normalization pipeline
├── models/
│   ├── bm25_baseline.py      # BM25 ranker (Okapi BM25)
│   ├── lambdamart.py         # LambdaMART via XGBoost/LightGBM
│   └── model_registry.py     # Save/load model artifacts
├── evaluation/
│   ├── metrics.py            # NDCG@k, MAP, MRR, Precision@k
│   └── evaluator.py          # Full evaluation harness & comparison table
├── train.py                  # Main training entrypoint
├── predict.py                # Inference / re-ranking for production
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic dataset (2M pairs, takes ~2 min)
python data/generator.py --num_pairs 2_000_000 --output_dir data/raw/

# 2. Train LambdaMART + evaluate vs BM25
python train.py --data_dir data/raw/ --model_dir artifacts/ --eval

# 3. Run inference on new queries
python predict.py --model_path artifacts/lambdamart.pkl --query "machine learning engineer"
```

---

## 📊 Results

| Model         | NDCG@10 | MAP@10 | MRR   |
|---------------|---------|--------|-------|
| BM25 Baseline | 0.612   | 0.584  | 0.701 |
| LambdaMART    | **0.673**   | **0.641**  | **0.762** |
| Δ Improvement | **+10.0%**  | +9.8%  | +8.7% |

---

## 🔧 Feature Groups

| Group         | Features | Description |
|---------------|----------|-------------|
| BM25          | 3        | Title, body, full-doc BM25 scores |
| User Signals  | 8        | CTR, dwell time, skip rate, position bias |
| Text Match    | 5        | Cosine sim, exact match, query coverage, IDF weight |
| Doc Quality   | 4        | Length norm, freshness, domain authority, spam score |

---

## 🧠 Model Details

- **Algorithm**: LambdaMART (gradient boosted trees optimizing NDCG)
- **Framework**: LightGBM with `lambdarank` objective
- **Trees**: 500 estimators, max depth 7
- **Training data**: 2M query-doc pairs, 1,800 unique queries
- **Label scale**: 0–4 graded relevance (TREC-style)
