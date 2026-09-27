import numpy as np
import pandas as pd
import pytest

import keyword_classify
from keyword_classify import classify, load_categories

# ---- load_categories ---------------------------------------------------------

def test_load_categories_parses_hints_and_threshold(tmp_path):
    path = tmp_path / "categories.yaml"
    path.write_text(
        """
unknown_threshold: 0.42
categories:
  CatA:
    hints:
      - hint one
      - hint two
  CatB:
    hints:
      - hint three
"""
    )
    categories, threshold = load_categories(str(path))
    assert categories == {"CatA": ["hint one", "hint two"], "CatB": ["hint three"]}
    assert threshold == pytest.approx(0.42)


def test_load_categories_default_threshold_when_omitted(tmp_path):
    path = tmp_path / "categories.yaml"
    path.write_text(
        """
categories:
  CatA:
    hints: []
"""
    )
    _, threshold = load_categories(str(path))
    assert threshold == pytest.approx(0.35)


def test_load_categories_handles_missing_hints_key(tmp_path):
    path = tmp_path / "categories.yaml"
    path.write_text(
        """
categories:
  CatA: {}
"""
    )
    categories, _ = load_categories(str(path))
    assert categories == {"CatA": []}


# ---- classify -----------------------------------------------------------------

VECTOR_MAP = {
    "CatA": [1.0, 0.0],
    "hint_a": [1.0, 0.0],
    "CatB": [0.0, 1.0],
    "hint_b": [0.0, 1.0],
    "kw_matches_a": [1.0, 0.0],
    "kw_matches_b": [0.0, 1.0],
    "kw_low_conf": [-1.0, 0.0],  # anti-correlated with CatA, orthogonal to CatB
}


@pytest.fixture
def fake_embed(monkeypatch):
    def fake_embed_keywords(terms, model_name, device):
        return np.array([VECTOR_MAP[t] for t in terms], dtype=np.float32)

    monkeypatch.setattr(keyword_classify, "embed_keywords", fake_embed_keywords)


def _classify_fixture_df():
    return pd.DataFrame(
        {
            "keyword": ["kw_matches_a", "kw_matches_b", "kw_low_conf"],
            "cpc": [1.0, 2.0, 3.0],
        }
    )


def test_classify_assigns_nearest_category(fake_embed):
    categories = {"CatA": ["hint_a"], "CatB": ["hint_b"]}
    result = classify(_classify_fixture_df(), categories, unknown_threshold=0.35, model_name="fake", device="cpu")

    a_row = result[result["keyword"] == "kw_matches_a"].iloc[0]
    b_row = result[result["keyword"] == "kw_matches_b"].iloc[0]
    assert a_row["category"] == "CatA"
    assert a_row["category_sim"] == pytest.approx(1.0)
    assert b_row["category"] == "CatB"
    assert b_row["category_sim"] == pytest.approx(1.0)


def test_classify_falls_back_to_unknown_below_threshold(fake_embed):
    categories = {"CatA": ["hint_a"], "CatB": ["hint_b"]}
    result = classify(_classify_fixture_df(), categories, unknown_threshold=0.35, model_name="fake", device="cpu")

    low_row = result[result["keyword"] == "kw_low_conf"].iloc[0]
    assert low_row["category"] == "Unknown"
    assert low_row["closest_category"] == "CatB"  # best match kept even though below threshold
    assert low_row["category_sim"] == pytest.approx(0.0)


def test_classify_unknown_threshold_is_configurable(fake_embed):
    categories = {"CatA": ["hint_a"], "CatB": ["hint_b"]}
    # with threshold below the achieved similarity (0.0), the same keyword should NOT be Unknown
    result = classify(_classify_fixture_df(), categories, unknown_threshold=-1.0, model_name="fake", device="cpu")
    low_row = result[result["keyword"] == "kw_low_conf"].iloc[0]
    assert low_row["category"] == "CatB"
