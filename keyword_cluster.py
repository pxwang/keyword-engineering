#!/usr/bin/env python3
"""Cluster keyword terms by semantic similarity and rank by similarity * CPC.

Usage:
    python keyword_cluster.py --input data/keywords_sample.csv --threshold 0.78
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import util

from keyword_classify import classify, load_categories
from keyword_common import embed_keywords, load_keywords, pick_device


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def cluster_by_threshold(embeddings: np.ndarray, threshold: float) -> tuple[list[int], np.ndarray]:
    """Group terms into connected components wherever pairwise cosine similarity >= threshold."""
    n = embeddings.shape[0]
    sim = util.cos_sim(embeddings, embeddings).numpy()
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                uf.union(i, j)

    root_to_group: dict[int, int] = {}
    labels = []
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        labels.append(root_to_group[root])
    return labels, sim


def rank_terms(df: pd.DataFrame, sim: np.ndarray) -> pd.DataFrame:
    """Rank terms by similarity_to_representative * cpc; rank groups by average cpc.

    Average, not sum: a cluster's members are near-duplicate phrasings of the same
    search intent, not independent revenue streams, so summing cpc across them just
    rewards clusters for having more members rather than reflecting real value.
    """
    df = df.copy()
    df["group_size"] = df.groupby("group")["keyword"].transform("count")
    df["group_cpc_avg"] = df.groupby("group")["cpc"].transform("mean")

    # representative term per group = highest-CPC member
    rep_row_idx = df.groupby("group")["cpc"].idxmax()
    rep_index_by_group = {df.loc[i, "group"]: i for i in rep_row_idx}

    similarity_to_rep = np.empty(len(df))
    for row_i, group in zip(df.index, df["group"], strict=True):
        similarity_to_rep[row_i] = sim[row_i, rep_index_by_group[group]]

    df["similarity_to_rep"] = similarity_to_rep
    df["score"] = df["similarity_to_rep"] * df["cpc"]
    df["representative_term"] = df["group"].map(
        lambda g: df.loc[rep_index_by_group[g], "keyword"]
    )

    return df.sort_values(["group_cpc_avg", "score"], ascending=False).reset_index(drop=True)


def write_report(df: pd.DataFrame, path: str, min_group_size: int = 1) -> None:
    """Write a grouped, human-readable markdown report: one section per cluster,
    sorted by group_cpc_avg, with members listed under their representative term.

    min_group_size filters out small/singleton groups that aren't really "clusters" -
    they still appear in the CSV output, just not in this report.
    """
    shown = df.drop_duplicates("group")
    shown = shown[shown["group_size"] >= min_group_size]
    skipped = df["group"].nunique() - len(shown)

    lines = [f"# Keyword clusters ({len(shown)} groups shown, {df['group'].nunique()} total, {len(df)} keywords)\n"]
    if skipped:
        lines.append(f"_{skipped} group(s) with fewer than {min_group_size} members omitted from this report._\n")

    group_order = shown.sort_values("group_cpc_avg", ascending=False)["group"].tolist()

    for rank, group in enumerate(group_order, start=1):
        members = df[df["group"] == group].sort_values("score", ascending=False)
        rep = members.iloc[0]["representative_term"]
        size = int(members.iloc[0]["group_size"])
        avg = members.iloc[0]["group_cpc_avg"]
        lines.append(f"## {rank}. {rep} — {size} keyword{'s' if size != 1 else ''}, ${avg:,.2f} avg cpc\n")
        for _, row in members.iterrows():
            marker = "**" if row["keyword"] == rep else ""
            lines.append(
                f"- {marker}{row['keyword']}{marker} "
                f"(cpc=${row['cpc']:.2f}, sim={row['similarity_to_rep']:.2f}, score={row['score']:.2f})"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def cluster_within_categories(
    df: pd.DataFrame, threshold: float, model_name: str, device: str
) -> pd.DataFrame:
    """Run the exact clustering+ranking pipeline independently within each 'category'
    bucket, then stitch the results back into one globally-ranked table.

    Requires df to already have a 'category' column (see keyword_classify.classify).
    Keywords in different categories can never end up in the same group - that's the
    whole point: it turns one O(n^2) clustering problem into several much smaller
    ones, at the cost of being unable to catch a legitimate cross-category match
    (e.g. "engagement rings" clustering with "wedding rings" across a shopping/wedding
    boundary won't happen in this mode, since they're classified into different
    buckets before clustering ever runs).
    """
    ranked_parts = []
    next_group_id = 0

    for category, bucket in df.groupby("category", sort=False):
        bucket = bucket.reset_index(drop=True)
        embeddings = embed_keywords(bucket["keyword"].tolist(), model_name, device)
        labels, sim = cluster_by_threshold(embeddings, threshold)
        bucket["group"] = [label + next_group_id for label in labels]
        next_group_id += max(labels) + 1

        ranked_bucket = rank_terms(bucket, sim)
        ranked_parts.append(ranked_bucket)
        print(f"  {category}: {len(bucket)} keywords -> {len(set(labels))} groups", file=sys.stderr)

    ranked = pd.concat(ranked_parts, ignore_index=True)
    return ranked.sort_values(["group_cpc_avg", "score"], ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/keywords_sample.csv", help="CSV with 'keyword' and 'cpc' columns")
    parser.add_argument("--output", default="output/keyword_groups.csv", help="Where to write the ranked CSV")
    parser.add_argument(
        "--report",
        default="output/keyword_report.md",
        help="Where to write a human-readable grouped markdown report (set to '' to skip)",
    )
    parser.add_argument(
        "--report-min-size",
        type=int,
        default=1,
        help="Omit groups smaller than this from the report (default 1 = show everything; try 2 to hide singletons)",
    )
    parser.add_argument("--threshold", type=float, default=0.78, help="Cosine similarity cutoff for grouping terms")
    parser.add_argument(
        "--cpc-weight",
        type=float,
        default=0.5,
        help="Where to sample cpc within [cpc_low, cpc_high]: 0.0=low bid, 1.0=high bid, 0.5=midpoint (default)",
    )
    parser.add_argument(
        "--classify-first",
        action="store_true",
        help="Classify into coarse industry categories first (see keyword_classify.py), then cluster "
        "independently within each category. Use for large keyword sets where clustering the whole "
        "dataset at once would be too slow/memory-heavy (see README's scalability notes).",
    )
    parser.add_argument("--categories", default="config/categories.yaml", help="YAML file defining categories and hint phrases (only used with --classify-first)")
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=None,
        help="Override the YAML file's unknown_threshold (only used with --classify-first)",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-transformers model name")
    parser.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto-detect, preferring GPU)")
    args = parser.parse_args()

    device = args.device or pick_device()
    print(f"Using device: {device}", file=sys.stderr)

    df = load_keywords(args.input, args.cpc_weight)

    if args.classify_first:
        categories, yaml_threshold = load_categories(args.categories)
        unknown_threshold = args.unknown_threshold if args.unknown_threshold is not None else yaml_threshold
        df = classify(df, categories, unknown_threshold, args.model, device)
        print(f"Classified {len(df)} keywords into {df['category'].nunique()} categories:", file=sys.stderr)
        ranked = cluster_within_categories(df, args.threshold, args.model, device)
    else:
        embeddings = embed_keywords(df["keyword"].tolist(), args.model, device)
        labels, sim = cluster_by_threshold(embeddings, args.threshold)
        df["group"] = labels
        ranked = rank_terms(df, sim)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ranked.to_csv(args.output, index=False)

    n_groups = ranked["group"].nunique()
    print(f"{len(df)} keywords -> {n_groups} groups (threshold={args.threshold})", file=sys.stderr)
    preview_cols = ["group", "representative_term", "keyword", "cpc", "similarity_to_rep", "score", "group_size"]
    print(ranked[preview_cols].to_string(index=False), file=sys.stderr)
    print(f"\nWrote {args.output}", file=sys.stderr)

    if args.report:
        report_dir = os.path.dirname(args.report)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        write_report(ranked, args.report, args.report_min_size)
        print(f"Wrote {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
