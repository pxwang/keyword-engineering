#!/usr/bin/env python3
"""Shared loading/embedding primitives used by keyword_cluster.py, keyword_peers.py,
and keyword_classify.py. Kept in its own module so those three can import from a
common place without any risk of circular imports between them."""
import os
import sqlite3

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_CACHE_PATH = "cache/embeddings.db"
_SQLITE_MAX_PARAMS = 500  # stay well under SQLite's variable-count limit when batching lookups


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_keywords(path: str, cpc_weight: float) -> pd.DataFrame:
    """Load keywords and derive a single cpc value.

    Supports two input shapes:
    - raw bid range: 'cpc_low' + 'cpc_high' columns -> cpc = low + (high - low) * cpc_weight
      (cpc_weight=0.0 picks the low bid, 1.0 the high bid, 0.5 the midpoint)
    - precomputed: a single 'cpc' column, used as-is (cpc_weight is ignored)
    """
    df = pd.read_csv(path)
    if "keyword" not in df.columns:
        raise ValueError("Input CSV must have a 'keyword' column")

    if {"cpc_low", "cpc_high"}.issubset(df.columns):
        df["cpc_low"] = df["cpc_low"].astype(float)
        df["cpc_high"] = df["cpc_high"].astype(float)
        df["cpc"] = df["cpc_low"] + (df["cpc_high"] - df["cpc_low"]) * cpc_weight
    elif "cpc" in df.columns:
        df["cpc"] = df["cpc"].astype(float)
    else:
        raise ValueError("Input CSV must have either 'cpc_low'/'cpc_high' or a 'cpc' column")

    df = df.dropna(subset=["keyword", "cpc"]).drop_duplicates(subset="keyword")
    return df.reset_index(drop=True)


def _open_cache(cache_path: str) -> sqlite3.Connection:
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            model TEXT NOT NULL,
            keyword TEXT NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (model, keyword)
        )
        """
    )
    return conn


def _fetch_cached(conn: sqlite3.Connection, model_name: str, terms: list[str]) -> dict[str, np.ndarray]:
    """Look up whichever of `terms` already have a cached embedding for model_name."""
    cached: dict[str, np.ndarray] = {}
    unique_terms = list(dict.fromkeys(terms))  # de-dup while preserving order
    for i in range(0, len(unique_terms), _SQLITE_MAX_PARAMS):
        chunk = unique_terms[i : i + _SQLITE_MAX_PARAMS]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT keyword, embedding FROM embeddings WHERE model = ? AND keyword IN ({placeholders})",
            [model_name, *chunk],
        ).fetchall()
        for keyword, blob in rows:
            cached[keyword] = np.frombuffer(blob, dtype=np.float32)
    return cached


def embed_keywords(
    terms: list[str],
    model_name: str,
    device: str,
    cache_path: str | None = DEFAULT_CACHE_PATH,
) -> np.ndarray:
    """Embed a list of keyword strings, reusing a persistent on-disk cache keyed by
    (model_name, keyword) so repeat keywords - across scripts, across separate runs,
    or within one --classify-first run that embeds the same keywords more than once -
    never get re-encoded by the model.

    Pass cache_path=None to disable caching and always call the model directly.
    """
    if cache_path is None:
        model = SentenceTransformer(model_name, device=device)
        return model.encode(terms, normalize_embeddings=True, show_progress_bar=False)

    conn = _open_cache(cache_path)
    try:
        cached = _fetch_cached(conn, model_name, terms)
        missing = [t for t in dict.fromkeys(terms) if t not in cached]

        if missing:
            model = SentenceTransformer(model_name, device=device)
            fresh = model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model, keyword, embedding) VALUES (?, ?, ?)",
                [
                    (model_name, term, emb.astype(np.float32).tobytes())
                    for term, emb in zip(missing, fresh, strict=True)
                ],
            )
            conn.commit()
            cached.update(zip(missing, fresh, strict=True))

        return np.stack([cached[t] for t in terms])
    finally:
        conn.close()
