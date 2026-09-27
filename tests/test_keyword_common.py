import numpy as np
import pandas as pd
import pytest

import keyword_common
from keyword_common import embed_keywords, load_keywords


# ---- load_keywords ---------------------------------------------------------

def test_load_keywords_cpc_range_midpoint(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("keyword,cpc_low,cpc_high\na,1.0,3.0\n")
    df = load_keywords(str(path), cpc_weight=0.5)
    assert df.loc[0, "cpc"] == pytest.approx(2.0)


def test_load_keywords_cpc_range_low_and_high_bounds(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("keyword,cpc_low,cpc_high\na,1.0,3.0\n")
    low = load_keywords(str(path), cpc_weight=0.0)
    high = load_keywords(str(path), cpc_weight=1.0)
    assert low.loc[0, "cpc"] == pytest.approx(1.0)
    assert high.loc[0, "cpc"] == pytest.approx(3.0)


def test_load_keywords_precomputed_cpc_used_as_is(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("keyword,cpc\na,4.2\n")
    df = load_keywords(str(path), cpc_weight=0.5)  # weight must be ignored here
    assert df.loc[0, "cpc"] == pytest.approx(4.2)


def test_load_keywords_missing_keyword_column_raises(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("term,cpc\na,1.0\n")
    with pytest.raises(ValueError):
        load_keywords(str(path), cpc_weight=0.5)


def test_load_keywords_missing_cpc_columns_raises(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("keyword,search_volume\na,100\n")
    with pytest.raises(ValueError):
        load_keywords(str(path), cpc_weight=0.5)


def test_load_keywords_dedups_and_drops_missing_cpc(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("keyword,cpc\na,1.0\na,1.0\nb,\n")
    df = load_keywords(str(path), cpc_weight=0.5)
    assert list(df["keyword"]) == ["a"]


# ---- embed_keywords cache ---------------------------------------------------

class FakeSentenceTransformer:
    """Records every batch of terms it's asked to encode, so tests can assert
    whether the (fake, expensive) model was actually called or the cache
    served the result instead."""

    encode_calls: list[list[str]] = []

    def __init__(self, model_name, device):
        self.model_name = model_name

    def encode(self, terms, normalize_embeddings=True, show_progress_bar=False):
        FakeSentenceTransformer.encode_calls.append(list(terms))
        # deterministic per-term vector, no real model involved
        return np.array([[float(len(t)), float(sum(map(ord, t)) % 97)] for t in terms], dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_fake_calls():
    FakeSentenceTransformer.encode_calls = []
    yield


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(keyword_common, "SentenceTransformer", FakeSentenceTransformer)


def test_embed_keywords_cache_miss_then_hit(fake_model, tmp_path):
    cache_path = str(tmp_path / "cache.db")

    first = embed_keywords(["alpha", "beta"], "fake-model", "cpu", cache_path=cache_path)
    assert len(FakeSentenceTransformer.encode_calls) == 1
    assert FakeSentenceTransformer.encode_calls[0] == ["alpha", "beta"]

    second = embed_keywords(["alpha", "beta"], "fake-model", "cpu", cache_path=cache_path)
    # cache hit: model must NOT be called again
    assert len(FakeSentenceTransformer.encode_calls) == 1
    np.testing.assert_array_equal(first, second)


def test_embed_keywords_partial_cache_hit_only_embeds_missing(fake_model, tmp_path):
    cache_path = str(tmp_path / "cache.db")

    embed_keywords(["alpha", "beta"], "fake-model", "cpu", cache_path=cache_path)
    embed_keywords(["beta", "gamma"], "fake-model", "cpu", cache_path=cache_path)

    assert len(FakeSentenceTransformer.encode_calls) == 2
    assert FakeSentenceTransformer.encode_calls[1] == ["gamma"]  # only the uncached term


def test_embed_keywords_cache_path_none_always_calls_model(fake_model, tmp_path):
    embed_keywords(["alpha"], "fake-model", "cpu", cache_path=None)
    embed_keywords(["alpha"], "fake-model", "cpu", cache_path=None)
    assert len(FakeSentenceTransformer.encode_calls) == 2


def test_embed_keywords_cache_is_keyed_by_model(fake_model, tmp_path):
    cache_path = str(tmp_path / "cache.db")
    embed_keywords(["alpha"], "model-a", "cpu", cache_path=cache_path)
    embed_keywords(["alpha"], "model-b", "cpu", cache_path=cache_path)
    # same keyword, different model -> both must hit the (fake) model
    assert len(FakeSentenceTransformer.encode_calls) == 2
