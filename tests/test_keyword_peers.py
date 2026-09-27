import numpy as np
import pandas as pd
import pytest

from keyword_peers import find_peers, sort_by_best_peer


def _make_df_and_sim():
    keywords = ["a", "b", "c", "d", "e"]
    cpc = [10.0, 20.0, 5.0, 100.0, 50.0]
    df = pd.DataFrame({"keyword": keywords, "cpc": cpc})

    # only row "a" is exercised in detail by the tests below:
    #   b: cpc 20 (+100%), sim 0.6  -> qualifies at premium=0.3, min_sim=0.5
    #   c: cpc  5 (-50%),  sim 0.9  -> excluded: cpc too low despite high similarity
    #   d: cpc 100 (+900%), sim 0.4 -> excluded: similarity too low despite huge cpc
    #   e: cpc 50 (+400%), sim 0.55 -> qualifies
    sim = np.array(
        [
            [1.00, 0.60, 0.90, 0.40, 0.55],
            [0.60, 1.00, 0.30, 0.20, 0.35],
            [0.90, 0.30, 1.00, 0.10, 0.20],
            [0.40, 0.20, 0.10, 1.00, 0.30],
            [0.55, 0.35, 0.20, 0.30, 1.00],
        ]
    )
    return df, sim


def test_find_peers_applies_cpc_premium_filter():
    df, sim = _make_df_and_sim()
    peers = find_peers(df, sim, cpc_premium=0.3, cpc_weight=0.0, top_n=5, min_sim=0.0)
    a_peers = set(peers[peers["keyword"] == "a"]["peer_keyword"])
    assert "c" not in a_peers  # lower cpc than a, excluded despite sim=0.9


def test_find_peers_applies_min_sim_filter():
    df, sim = _make_df_and_sim()
    peers = find_peers(df, sim, cpc_premium=0.3, cpc_weight=0.0, top_n=5, min_sim=0.5)
    a_peers = set(peers[peers["keyword"] == "a"]["peer_keyword"])
    assert a_peers == {"b", "e"}  # d excluded: sim 0.4 < 0.5, despite huge cpc


def test_find_peers_never_returns_self_as_peer():
    df, sim = _make_df_and_sim()
    peers = find_peers(df, sim, cpc_premium=0.3, cpc_weight=0.0, top_n=5, min_sim=0.0)
    assert not (peers["keyword"] == peers["peer_keyword"]).any()


def test_find_peers_top_n_truncates():
    df, sim = _make_df_and_sim()
    peers = find_peers(df, sim, cpc_premium=0.3, cpc_weight=0.0, top_n=1, min_sim=0.5)
    a_peers = peers[peers["keyword"] == "a"]
    assert len(a_peers) == 1
    assert a_peers.iloc[0]["peer_keyword"] == "b"  # higher sim_score (0.6 > 0.55) wins at cpc_weight=0


def test_find_peers_cpc_weight_changes_ranking():
    df, sim = _make_df_and_sim()
    # cpc_weight=0: b (sim 0.6) outranks e (sim 0.55)
    low_weight = find_peers(df, sim, cpc_premium=0.3, cpc_weight=0.0, top_n=5, min_sim=0.5)
    a_low = low_weight[low_weight["keyword"] == "a"].sort_values("peer_rank")
    assert list(a_low["peer_keyword"]) == ["b", "e"]

    # a large enough cpc_weight should let e's much higher cpc (50 vs 20) overtake
    high_weight = find_peers(df, sim, cpc_premium=0.3, cpc_weight=0.02, top_n=5, min_sim=0.5)
    a_high = high_weight[high_weight["keyword"] == "a"].sort_values("peer_rank")
    assert list(a_high["peer_keyword"]) == ["e", "b"]


def test_find_peers_score_formula():
    df, sim = _make_df_and_sim()
    peers = find_peers(df, sim, cpc_premium=0.3, cpc_weight=0.1, top_n=5, min_sim=0.5)
    b_row = peers[(peers["keyword"] == "a") & (peers["peer_keyword"] == "b")].iloc[0]
    assert b_row["score"] == pytest.approx(20.0 * 0.1 + 0.6)


def test_find_peers_cpc_uplift_computed_correctly():
    df, sim = _make_df_and_sim()
    peers = find_peers(df, sim, cpc_premium=0.3, cpc_weight=0.0, top_n=5, min_sim=0.5)
    b_row = peers[(peers["keyword"] == "a") & (peers["peer_keyword"] == "b")].iloc[0]
    assert b_row["cpc_uplift"] == pytest.approx(20.0 / 10.0 - 1)  # +100%


def test_find_peers_keyword_with_no_qualifying_peers_is_absent():
    df, sim = _make_df_and_sim()
    peers = find_peers(df, sim, cpc_premium=0.3, cpc_weight=0.0, top_n=5, min_sim=0.5)
    # "d" has the highest cpc (100) - nothing is 30% higher, so it should have no rows
    assert "d" not in set(peers["keyword"])


def test_sort_by_best_peer_orders_by_rank1_score_descending():
    df = pd.DataFrame(
        {
            "keyword": ["low", "low", "high"],
            "peer_rank": [1, 2, 1],
            "score": [0.3, 0.1, 0.9],
        }
    )
    sorted_df = sort_by_best_peer(df)
    assert list(sorted_df["keyword"]) == ["high", "low", "low"]
