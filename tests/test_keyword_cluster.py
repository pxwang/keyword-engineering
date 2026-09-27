import math

import numpy as np
import pandas as pd
import pytest

import keyword_cluster
from keyword_cluster import (
    UnionFind,
    cluster_by_threshold,
    cluster_within_categories,
    rank_terms,
    write_report,
)


# ---- UnionFind ---------------------------------------------------------

def test_union_find_basic():
    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(2, 3)
    assert uf.find(0) == uf.find(1)
    assert uf.find(2) == uf.find(3)
    assert uf.find(0) != uf.find(2)


def test_union_find_transitive_chain():
    uf = UnionFind(3)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)  # chained through 1, even though never unioned directly


# ---- cluster_by_threshold ------------------------------------------------

def _unit_vector(angle_degrees: float) -> list[float]:
    rad = math.radians(angle_degrees)
    return [math.cos(rad), math.sin(rad)]


def test_cluster_by_threshold_groups_close_pairs_and_separates_far_ones():
    # v0/v1 are 10 degrees apart (cos=0.985, above threshold); v2/v3 are 5 degrees
    # apart (cos=0.996, above threshold); the two pairs are ~80-95 degrees apart
    # from each other (well below threshold) - so we expect exactly two groups.
    embeddings = np.array(
        [_unit_vector(0), _unit_vector(10), _unit_vector(90), _unit_vector(95)],
        dtype=np.float32,
    )
    labels, sim = cluster_by_threshold(embeddings, threshold=0.78)

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert len(set(labels)) == 2


def test_cluster_by_threshold_all_singletons_when_nothing_similar_enough():
    embeddings = np.array(
        [_unit_vector(0), _unit_vector(90), _unit_vector(180)],
        dtype=np.float32,
    )
    labels, _ = cluster_by_threshold(embeddings, threshold=0.99)
    assert len(set(labels)) == 3


# ---- rank_terms ------------------------------------------------------------

def test_rank_terms_uses_average_not_sum_for_group_cpc():
    df = pd.DataFrame({"keyword": ["a", "b", "c"], "cpc": [10.0, 5.0, 100.0], "group": [0, 0, 1]})
    sim = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ranked = rank_terms(df, sim)

    group0 = ranked[ranked["group"] == 0]
    assert group0["group_cpc_avg"].iloc[0] == pytest.approx(7.5)  # mean(10, 5), not sum(15)
    assert group0["group_size"].iloc[0] == 2


def test_rank_terms_representative_is_highest_cpc_in_group():
    df = pd.DataFrame({"keyword": ["a", "b"], "cpc": [10.0, 5.0], "group": [0, 0]})
    sim = np.array([[1.0, 0.9], [0.9, 1.0]])
    ranked = rank_terms(df, sim)
    assert (ranked["representative_term"] == "a").all()


def test_rank_terms_score_is_similarity_times_cpc():
    df = pd.DataFrame({"keyword": ["a", "b"], "cpc": [10.0, 5.0], "group": [0, 0]})
    sim = np.array([[1.0, 0.9], [0.9, 1.0]])
    ranked = rank_terms(df, sim)
    b_row = ranked[ranked["keyword"] == "b"].iloc[0]
    assert b_row["similarity_to_rep"] == pytest.approx(0.9)
    assert b_row["score"] == pytest.approx(0.9 * 5.0)


def test_rank_terms_sorts_by_group_cpc_avg_then_score():
    df = pd.DataFrame({"keyword": ["a", "b", "c"], "cpc": [10.0, 5.0, 100.0], "group": [0, 0, 1]})
    sim = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ranked = rank_terms(df, sim)
    # group 1 (avg cpc 100) must lead, then group 0 sorted a (score 10) before b (score 4.5)
    assert list(ranked["keyword"]) == ["c", "a", "b"]


# ---- write_report -----------------------------------------------------------

def test_write_report_min_group_size_hides_singletons(tmp_path):
    df = pd.DataFrame({"keyword": ["a", "b", "c"], "cpc": [10.0, 5.0, 100.0], "group": [0, 0, 1]})
    sim = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ranked = rank_terms(df, sim)
    out_path = tmp_path / "report.md"

    write_report(ranked, str(out_path), min_group_size=2)
    text = out_path.read_text()

    assert "1 groups shown, 2 total" in text
    assert "- a" in text or "**a**" in text  # group 0 (size 2) shown
    assert "c " not in text.split("\n")[0]  # singleton group (c) omitted from the body


# ---- cluster_within_categories ----------------------------------------------

def test_cluster_within_categories_never_merges_across_categories(monkeypatch):
    vectors = {
        "cat-a-1": [1.0, 0.0],
        "cat-a-2": [0.98, 0.02],  # close to cat-a-1, should cluster with it
        "cat-b-1": [0.0, 1.0],
        "cat-b-2": [0.02, 0.98],  # close to cat-b-1, should cluster with it
    }

    def fake_embed_keywords(terms, model_name, device):
        return np.array([vectors[t] for t in terms], dtype=np.float32)

    monkeypatch.setattr(keyword_cluster, "embed_keywords", fake_embed_keywords)

    df = pd.DataFrame(
        {
            "keyword": list(vectors.keys()),
            "cpc": [10.0, 8.0, 20.0, 15.0],
            "category": ["A", "A", "B", "B"],
        }
    )

    ranked = cluster_within_categories(df, threshold=0.9, model_name="fake", device="cpu")

    # no group should span both categories
    mixed = ranked.groupby("group")["category"].nunique()
    assert (mixed <= 1).all()

    # group ids must be globally unique (no collision between the two per-category runs)
    a_groups = set(ranked[ranked["category"] == "A"]["group"])
    b_groups = set(ranked[ranked["category"] == "B"]["group"])
    assert a_groups.isdisjoint(b_groups)
