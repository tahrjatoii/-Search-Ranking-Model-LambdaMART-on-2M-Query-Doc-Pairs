"""
data/generator.py
-----------------
Generates a realistic synthetic dataset of 2M+ query-document pairs
with graded relevance labels (0-4, TREC-style), BM25-scorable text,
and user engagement signals.

Usage:
    python data/generator.py --num_pairs 2_000_000 --output_dir data/raw/
"""

import argparse
import os
import random
import math
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Domain vocabulary ────────────────────────────────────────────────────────

QUERY_TEMPLATES = [
    "{skill} engineer {level}",
    "{skill} developer remote",
    "{role} {location} salary",
    "senior {skill} {role}",
    "{role} with {skill} experience",
    "{skill} {role} job",
    "junior {skill} engineer",
    "lead {skill} architect",
    "{role} machine learning",
    "data {role} {skill}",
]

SKILLS = [
    "python", "java", "golang", "rust", "kubernetes", "react", "pytorch",
    "tensorflow", "spark", "kafka", "postgres", "redis", "aws", "gcp",
    "azure", "docker", "mlops", "llm", "nlp", "computer vision",
    "recommendation systems", "distributed systems", "system design",
    "backend", "frontend", "fullstack", "devops", "platform", "security",
]

ROLES = [
    "engineer", "developer", "scientist", "analyst", "architect",
    "manager", "lead", "principal", "staff", "director",
]

LEVELS = ["junior", "mid-level", "senior", "staff", "principal", "lead"]
LOCATIONS = ["new york", "san francisco", "remote", "seattle", "austin", "london"]

RESUME_SECTIONS = [
    "Built {skill} pipeline processing {n}M events/day reducing latency by {pct}%",
    "Led team of {n} engineers to migrate {role} infrastructure to {skill}",
    "Designed {skill} system achieving {nines} uptime with {n}K RPS",
    "Improved {metric} by {pct}% using {skill} optimization techniques",
    "Implemented {skill} model serving {n}M users with p99 latency < {ms}ms",
    "Architected {skill} data platform ingesting {n}TB/day",
    "Reduced cloud costs by {pct}% by re-architecting {skill} workloads",
    "Built real-time {skill} feature store used by {n} ML models",
    "Mentored {n} junior engineers on {skill} best practices",
    "Open-sourced {skill} library with {n}K GitHub stars",
]

COMPANIES = [
    "Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix",
    "Stripe", "Airbnb", "Uber", "Lyft", "Snowflake", "Databricks",
    "OpenAI", "Anthropic", "Cohere", "Startup", "MidSizeCo",
]

DEGREES = ["BS CS", "MS CS", "PhD ML", "BS EE", "MBA"]


def _rand_resume_bullet(skill: str) -> str:
    template = random.choice(RESUME_SECTIONS)
    return template.format(
        skill=skill,
        role=random.choice(ROLES),
        n=random.choice([2, 5, 10, 50, 100, 200]),
        pct=random.choice([10, 15, 20, 25, 30, 40, 50]),
        metric=random.choice(["NDCG@10", "precision", "recall", "latency", "throughput"]),
        nines=random.choice(["99.9%", "99.99%", "99.999%"]),
        ms=random.choice([10, 50, 100, 200]),
    )


def _generate_document(query_skills: list, relevance: int) -> dict:
    """
    Build a resume document. Higher relevance → more query skill overlap,
    better companies, more accomplishment bullets.
    """
    # How many query skills appear in the doc depends on relevance
    overlap_probs = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.8, 4: 1.0}
    doc_skills = []
    for s in query_skills:
        if random.random() < overlap_probs[relevance]:
            doc_skills.append(s)
    # Pad with random skills
    extra = random.sample(SKILLS, k=random.randint(2, 6))
    doc_skills = list(set(doc_skills + extra))

    num_bullets = {0: 1, 1: 2, 2: 3, 3: 5, 4: 7}[relevance]
    bullets = [_rand_resume_bullet(random.choice(doc_skills)) for _ in range(num_bullets)]

    years_exp = {0: random.randint(0, 1), 1: random.randint(1, 3),
                 2: random.randint(2, 5), 3: random.randint(4, 8), 4: random.randint(6, 15)}[relevance]

    company_tier = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}[relevance]
    start_idx = max(0, len(COMPANIES) - 4 * (company_tier + 1))
    company_pool = COMPANIES[start_idx:] if start_idx < len(COMPANIES) else COMPANIES
    company = random.choice(company_pool)

    title = " ".join([
        random.choice(LEVELS if relevance > 1 else LEVELS[:2]),
        random.choice(doc_skills[:2]) if doc_skills else random.choice(SKILLS),
        random.choice(ROLES),
    ])

    body = f"{title} at {company}. {years_exp} years experience. " + " ".join(bullets)
    return {
        "title": title,
        "body": body,
        "skills": doc_skills,
        "years_exp": years_exp,
        "company": company,
        "degree": random.choice(DEGREES),
        "num_bullets": num_bullets,
    }


def _generate_user_signals(relevance: int) -> dict:
    """
    Simulate user engagement signals correlated with relevance.
    Adds noise to simulate real-world messiness.
    """
    noise = lambda: random.gauss(0, 0.05)

    base_ctr     = [0.02, 0.05, 0.10, 0.20, 0.35][relevance]
    base_dwell   = [5,    15,   45,   90,   180][relevance]     # seconds
    base_skip    = [0.70, 0.50, 0.30, 0.15, 0.05][relevance]
    base_conv    = [0.01, 0.03, 0.07, 0.15, 0.30][relevance]

    ctr       = max(0.0, min(1.0, base_ctr   + noise()))
    dwell_sec = max(0.0, base_dwell * (1 + noise() * 2))
    skip_rate = max(0.0, min(1.0, base_skip  + noise()))
    conv_rate = max(0.0, min(1.0, base_conv  + noise()))
    impressions = random.randint(10, 5000)
    clicks      = max(0, int(impressions * ctr * (1 + random.gauss(0, 0.1))))

    return {
        "ctr":            round(ctr, 4),
        "dwell_time_sec": round(dwell_sec, 2),
        "skip_rate":      round(skip_rate, 4),
        "conversion_rate":round(conv_rate, 4),
        "impressions":    impressions,
        "clicks":         clicks,
        "avg_position":   round(random.uniform(1, 10), 2),
        "bookmark_rate":  round(max(0, conv_rate * 0.3 + noise() * 0.02), 4),
    }


def generate_query(template: str | None = None) -> tuple[str, list]:
    """Returns (query_text, extracted_skills)."""
    t = template or random.choice(QUERY_TEMPLATES)
    skill = random.choice(SKILLS)
    role  = random.choice(ROLES)
    level = random.choice(LEVELS)
    loc   = random.choice(LOCATIONS)
    query = t.format(skill=skill, role=role, level=level, location=loc)
    # Extract all skills mentioned in query
    skills_in_query = [s for s in SKILLS if s in query]
    if not skills_in_query:
        skills_in_query = [skill]
    return query, skills_in_query


def generate_dataset(
    num_pairs: int = 2_000_000,
    output_dir: str = "data/raw/",
    seed: int = 42,
    chunk_size: int = 100_000,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Relevance distribution (realistic: heavy tail at 0)
    rel_weights = [0.40, 0.25, 0.18, 0.12, 0.05]  # labels 0-4

    records = []
    query_id = 0
    doc_id   = 0
    chunk_n  = 0

    print(f"Generating {num_pairs:,} query-doc pairs...")
    pbar = tqdm(total=num_pairs, unit="pairs", ncols=80)

    # Group pairs by query (realistic: ~1100 docs per query on average)
    docs_per_query = max(1, num_pairs // 1800)

    pairs_generated = 0
    while pairs_generated < num_pairs:
        query_text, q_skills = generate_query()
        q_id = f"q{query_id:06d}"
        query_id += 1

        n_docs = random.randint(
            max(1, docs_per_query - 200),
            docs_per_query + 200
        )
        n_docs = min(n_docs, num_pairs - pairs_generated)

        for _ in range(n_docs):
            relevance = random.choices([0, 1, 2, 3, 4], weights=rel_weights)[0]
            doc = _generate_document(q_skills, relevance)
            signals = _generate_user_signals(relevance)

            record = {
                "query_id":  q_id,
                "doc_id":    f"d{doc_id:09d}",
                "query":     query_text,
                "relevance": relevance,
                # Document fields
                "doc_title": doc["title"],
                "doc_body":  doc["body"],
                "doc_skills":"|".join(doc["skills"]),
                "years_exp": doc["years_exp"],
                "num_bullets":doc["num_bullets"],
                "company":   doc["company"],
                "degree":    doc["degree"],
                # User signals
                **signals,
            }
            records.append(record)
            doc_id   += 1
            pairs_generated += 1
            pbar.update(1)

            # Flush chunk to disk
            if len(records) >= chunk_size:
                df = pd.DataFrame(records)
                df.to_parquet(
                    os.path.join(output_dir, f"chunk_{chunk_n:04d}.parquet"),
                    index=False,
                )
                records = []
                chunk_n += 1

    pbar.close()

    # Flush remainder
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(
            os.path.join(output_dir, f"chunk_{chunk_n:04d}.parquet"),
            index=False,
        )

    # Write metadata
    meta = {
        "num_pairs":    pairs_generated,
        "num_queries":  query_id,
        "num_docs":     doc_id,
        "num_chunks":   chunk_n + 1,
        "rel_labels":   [0, 1, 2, 3, 4],
        "seed":         seed,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Done! {pairs_generated:,} pairs written to {output_dir}")
    print(f"   Queries: {query_id:,}  |  Chunks: {chunk_n + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs",  type=int, default=2_000_000)
    parser.add_argument("--output_dir", type=str, default="data/raw/")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=100_000)
    args = parser.parse_args()
    generate_dataset(
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
