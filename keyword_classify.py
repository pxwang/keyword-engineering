#!/usr/bin/env python3
"""Classify each keyword into a coarse industry category by nearest-anchor similarity.

This is the "classify-first" stage of a bucket-then-cluster pipeline for large keyword
sets: each keyword gets assigned to whichever of a small fixed set of industry anchors
it's closest to (cosine similarity over the same all-MiniLM-L6-v2 embeddings used
elsewhere in this project), so exact O(n^2) clustering can then run per-category
instead of over the whole dataset at once.

The category names started from a real ad-industry taxonomy rather than being invented
from scratch, but the hint phrases, the "Unknown" fallback, and the confidence
threshold are this project's own additions - see config/categories.yaml.

Usage:
    python keyword_classify.py --input data/keyword_real.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import util

from keyword_common import embed_keywords, load_keywords, pick_device

UNKNOWN_LABEL = "Unknown"


def load_categories(path: str) -> tuple[dict[str, list[str]], float]:
    """Load the category -> hint-phrases map and default unknown_threshold from YAML.

    See config/categories.yaml for the schema and the reasoning behind hints and the
    unknown threshold.
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    categories = {name: spec.get("hints", []) for name, spec in config["categories"].items()}
    unknown_threshold = float(config.get("unknown_threshold", 0.35))
    return categories, unknown_threshold


def classify(
    df: pd.DataFrame,
    categories: dict[str, list[str]],
    unknown_threshold: float,
    model_name: str,
    device: str,
) -> pd.DataFrame:
    """Assign each keyword to its nearest anchor category, or 'Unknown' if even the
    best match falls below unknown_threshold.

    Each anchor's embedding is the mean of its label plus its configured hint phrases,
    not just the bare category label - a single short label doesn't carry enough signal
    to embed well against real keyword-style phrasing.
    """
    anchor_names = list(categories.keys())
    keyword_emb = embed_keywords(df["keyword"].tolist(), model_name, device)

    anchor_texts = []
    anchor_group_sizes = []
    for name in anchor_names:
        phrases = [name] + categories[name]
        anchor_texts.extend(phrases)
        anchor_group_sizes.append(len(phrases))

    anchor_phrase_emb = embed_keywords(anchor_texts, model_name, device)
    anchor_emb = np.empty((len(anchor_names), anchor_phrase_emb.shape[1]), dtype=anchor_phrase_emb.dtype)
    offset = 0
    for i, size in enumerate(anchor_group_sizes):
        group = anchor_phrase_emb[offset : offset + size]
        mean = group.mean(axis=0)
        anchor_emb[i] = mean / np.linalg.norm(mean)
        offset += size

    sim = util.cos_sim(keyword_emb, anchor_emb).numpy()  # (n_keywords, n_anchors)
    best_idx = np.argmax(sim, axis=1)
    best_sim = sim[np.arange(len(df)), best_idx]

    df = df.copy()
    df["closest_category"] = [anchor_names[i] for i in best_idx]
    df["category_sim"] = best_sim
    df["category"] = np.where(best_sim >= unknown_threshold, df["closest_category"], UNKNOWN_LABEL)
    return df.sort_values(["category", "cpc"], ascending=[True, False]).reset_index(drop=True)


def write_report(df: pd.DataFrame, path: str) -> None:
    """Grouped markdown report: one section per assigned category, keywords listed
    underneath sorted by cpc, so you can eyeball whether each bucket makes sense."""
    lines = [f"# Keyword categories ({df['category'].nunique()} categories used, {len(df)} keywords)\n"]

    sizes = df.groupby("category").size().sort_values(ascending=False)
    for category, size in sizes.items():
        group = df[df["category"] == category]
        avg_sim = group["category_sim"].mean()
        lines.append(f"## {category} — {size} keywords, avg confidence {avg_sim:.2f}\n")
        for _, row in group.iterrows():
            closest = (
                f", closest={row['closest_category']}" if category == UNKNOWN_LABEL else ""
            )
            lines.append(f"- {row['keyword']} (cpc=${row['cpc']:.2f}, sim={row['category_sim']:.2f}{closest})")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/keywords_sample.csv", help="CSV with 'keyword' and cpc columns")
    parser.add_argument("--output", default="output/keyword_categories.csv", help="Where to write the classified CSV")
    parser.add_argument("--report", default="output/keyword_categories_report.md", help="Human-readable grouped report ('' to skip)")
    parser.add_argument("--categories", default="config/categories.yaml", help="YAML file defining categories and hint phrases")
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=None,
        help="Override the YAML file's unknown_threshold: below this category_sim, category becomes 'Unknown'",
    )
    parser.add_argument(
        "--cpc-range-weight",
        type=float,
        default=0.5,
        help="Where to sample cpc within [cpc_low, cpc_high] for inputs using that format (default 0.5 = midpoint)",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-transformers model name")
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto-detect, preferring GPU)")
    args = parser.parse_args()

    device = args.device or pick_device()
    print(f"Using device: {device}", file=sys.stderr)

    categories, yaml_threshold = load_categories(args.categories)
    unknown_threshold = args.unknown_threshold if args.unknown_threshold is not None else yaml_threshold

    df = load_keywords(args.input, args.cpc_range_weight)
    classified = classify(df, categories, unknown_threshold, args.model, device)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    classified.to_csv(args.output, index=False)

    n_categories = classified["category"].nunique()
    print(f"{len(classified)} keywords -> {n_categories} categories used (unknown_threshold={unknown_threshold})", file=sys.stderr)
    print(classified.groupby("category").size().sort_values(ascending=False).to_string(), file=sys.stderr)
    print(f"\nWrote {args.output}", file=sys.stderr)

    if args.report:
        report_dir = os.path.dirname(args.report)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        write_report(classified, args.report)
        print(f"Wrote {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
