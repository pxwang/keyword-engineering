#!/usr/bin/env python3
"""For each keyword, find higher-value peer keywords worth targeting instead/alongside it.

A peer of keyword1 is any other keyword2 in the input whose cpc is at least
(1 + cpc_premium) times keyword1's cpc (default: 30% higher). Peers are ranked by
    score = cpc * cpc_weight + sim_score
where sim_score is cosine similarity between the two keywords' embeddings. With the
default cpc_weight=0, ranking is purely by similarity among the qualifying higher-cpc
candidates - i.e. "which higher-value keyword is closest to what I'm already targeting".

Usage:
    python keyword_peers.py --input data/keyword_real.csv --cpc-premium 0.3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import util

from keyword_common import embed_keywords, load_keywords, pick_device


def find_peers(
    df: pd.DataFrame,
    sim: np.ndarray,
    cpc_premium: float,
    cpc_weight: float,
    top_n: int,
    min_sim: float,
) -> pd.DataFrame:
    """For each keyword, rank candidates with cpc > cpc * (1 + cpc_premium) and
    sim_score >= min_sim by score = cpc * cpc_weight + sim_score, keeping the top_n.

    min_sim exists so a keyword with few or no genuine peers just gets fewer than
    top_n rows (or none) instead of being padded with irrelevant high-cpc keywords
    that merely happen to be "least dissimilar" among the candidate pool.
    """
    keywords = df["keyword"].to_numpy()
    cpcs = df["cpc"].to_numpy()
    n = len(df)

    rows = []
    for i in range(n):
        min_cpc = cpcs[i] * (1 + cpc_premium)
        candidate_idx = np.where((cpcs > min_cpc) & (sim[i] >= min_sim))[0]
        candidate_idx = candidate_idx[candidate_idx != i]
        if len(candidate_idx) == 0:
            continue

        sim_scores = sim[i, candidate_idx]
        scores = cpcs[candidate_idx] * cpc_weight + sim_scores
        order = np.argsort(-scores)[:top_n]

        for rank, k in enumerate(order, start=1):
            j = candidate_idx[k]
            rows.append(
                {
                    "keyword": keywords[i],
                    "cpc": cpcs[i],
                    "peer_rank": rank,
                    "peer_keyword": keywords[j],
                    "peer_cpc": cpcs[j],
                    "cpc_uplift": cpcs[j] / cpcs[i] - 1,
                    "sim_score": sim_scores[k],
                    "score": scores[k],
                }
            )

    return pd.DataFrame(rows)


def sort_by_best_peer(df: pd.DataFrame) -> pd.DataFrame:
    """Order rows by each keyword's best (rank-1) peer score descending, then by
    peer_rank within a keyword - so the strongest matches lead both the CSV and report."""
    keyword_order = (
        df[df["peer_rank"] == 1].sort_values("score", ascending=False)["keyword"].tolist()
    )
    df = df.copy()
    df["keyword"] = pd.Categorical(df["keyword"], categories=keyword_order, ordered=True)
    df = df.sort_values(["keyword", "peer_rank"]).reset_index(drop=True)
    df["keyword"] = df["keyword"].astype(str)
    return df


def write_report(df: pd.DataFrame, path: str) -> None:
    """Grouped markdown report: one section per keyword (in the order given), peers
    listed underneath. Call sort_by_best_peer(df) first to lead with strongest matches."""
    lines = [f"# Higher-value peer keywords ({df['keyword'].nunique()} keywords with peers)\n"]

    for keyword, group in df.groupby("keyword", sort=False):
        cpc = group.iloc[0]["cpc"]
        lines.append(f"## {keyword} (cpc=${cpc:.2f})\n")
        for _, row in group.sort_values("peer_rank").iterrows():
            lines.append(
                f"- {row['peer_keyword']} "
                f"(cpc=${row['peer_cpc']:.2f}, +{row['cpc_uplift']:.0%}, "
                f"sim={row['sim_score']:.2f}, score={row['score']:.2f})"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/keywords_sample.csv", help="CSV with 'keyword' and cpc columns")
    parser.add_argument("--output", default="output/keyword_peers.csv", help="Where to write the ranked CSV")
    parser.add_argument("--report", default="output/keyword_peers_report.md", help="Human-readable grouped report ('' to skip)")
    parser.add_argument(
        "--cpc-premium",
        type=float,
        default=0.3,
        help="Minimum cpc uplift required to count as a peer: peer_cpc > cpc * (1 + this) (default 0.3 = 30%%)",
    )
    parser.add_argument(
        "--cpc-weight",
        type=float,
        default=0.0,
        help="Weight on peer_cpc in the ranking score = peer_cpc * cpc_weight + sim_score (default 0.0 = rank by similarity alone)",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Max peers to keep per keyword (default 5)")
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.5,
        help="Minimum cosine similarity for a candidate to count as a peer at all (default 0.5)",
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

    df = load_keywords(args.input, args.cpc_range_weight)
    embeddings = embed_keywords(df["keyword"].tolist(), args.model, device)
    sim = util.cos_sim(embeddings, embeddings).numpy()

    peers = find_peers(df, sim, args.cpc_premium, args.cpc_weight, args.top_n, args.min_sim)
    peers = sort_by_best_peer(peers)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    peers.to_csv(args.output, index=False)

    n_with_peers = peers["keyword"].nunique()
    print(
        f"{len(df)} keywords -> {n_with_peers} have at least one peer "
        f"(cpc-premium={args.cpc_premium:.0%}, cpc-weight={args.cpc_weight}, top-n={args.top_n}, min-sim={args.min_sim})",
        file=sys.stderr,
    )
    print(f"Wrote {args.output}", file=sys.stderr)

    if args.report:
        report_dir = os.path.dirname(args.report)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        write_report(peers, args.report)
        print(f"Wrote {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
