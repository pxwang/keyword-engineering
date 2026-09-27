# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
source .venv/bin/activate      # all commands below assume this is active

ruff check .                   # lint (also runs in CI)
pytest                         # run all tests
pytest tests/test_keyword_cluster.py            # single test file
pytest tests/test_keyword_cluster.py::test_name # single test

python keyword_classify.py --input data/keyword_real.csv
python keyword_cluster.py --input data/keyword_real.csv --threshold 0.78
python keyword_peers.py --input data/keyword_real.csv
```

Tests are fast and fully deterministic — they use hand-crafted fixed vectors or a
monkeypatched fake model, never the real MiniLM model, so no download or GPU is
needed to run them. They check code correctness (union-find grouping, cpc
interpolation, filtering/sorting logic, cache hit/miss behavior), not result quality
on real data — that's what `EVALUATION.md` is for, and it is not part of the test
suite.

Ruff config lives in `pyproject.toml`: default rules plus import sorting (`I`) and
bugbear (`B`). `E501` (line-too-long) is deliberately disabled — several argparse
`help=` strings are long single lines by design.

## Architecture

Three independent CLI scripts (`keyword_classify.py`, `keyword_cluster.py`,
`keyword_peers.py`) share one pipeline via `keyword_common.py`, which exists purely
to avoid circular imports between the three:

- `load_keywords(path, cpc_weight)` — accepts either a `cpc_low`/`cpc_high` bid-range
  CSV (derives `cpc = low + (high - low) * cpc_weight`) or a precomputed `cpc` column
  (used as-is, weight ignored). Every script must handle both input shapes.
- `embed_keywords(terms, model_name, device, cache_path)` — the single chokepoint all
  three scripts call for embeddings. It transparently caches every embedding in a
  local SQLite DB at `cache/embeddings.db`, keyed by `(model_name, keyword)`, so a
  keyword is only ever run through the model once across scripts, runs, and repeated
  lookups within one `--classify-first` run. `cache/` is gitignored — it's local
  infra, not project data. Any new code path that needs embeddings should go through
  this function rather than calling `SentenceTransformer` directly, to stay inside
  the cache.
- `pick_device()` — auto-detects CUDA → MPS → CPU; every script exposes `--device` to
  override it.

Per-script logic:

- **`keyword_classify.py`** — assigns each keyword to the nearest category by cosine
  similarity against per-category anchor embeddings built from `config/categories.yaml`
  (category name + hand-written hint phrases, since a bare label embeds poorly against
  real keyword phrasing). Falls back to `"Unknown"` when the best match is below
  `unknown_threshold` (from the YAML, overridable via `--unknown-threshold`). Output
  keeps both `category` (possibly `"Unknown"`) and `closest_category` (best match
  regardless of confidence).
- **`keyword_cluster.py`** — connected-components clustering: joins two keywords
  whenever cosine similarity ≥ `--threshold`, transitively (this can chain: A~B~C
  groups even if A and C aren't directly similar — raising `--threshold` is the fix,
  not a code change). Ranks by `similarity_to_representative * cpc`. `--classify-first`
  runs `keyword_classify.py`'s logic first and clusters independently per category
  (avoids O(n²) blowup at scale, and guarantees no cluster spans categories) before
  stitching results back into one globally-ranked output — it does not fix
  within-category chaining, only cross-category merges.
- **`keyword_peers.py`** — for each keyword, finds *other* keywords that are both
  meaningfully higher-value (`peer.cpc > this.cpc * (1 + cpc_premium)`) and
  meaningfully related (`similarity >= min_sim`), ranked by
  `score = peer_cpc * cpc_weight + sim_score`. Has no `--classify-first` mode (searches
  the whole input regardless of category) — deliberate, not an oversight: see
  *Further work* in `README.md` before adding one.

Each script writes both a CSV (for further processing) and a human-readable Markdown
report (`--report`, pass `''` to skip) to `output/` (gitignored).

`config/categories.yaml` is the only place category taxonomy/hints/threshold should be
edited — no code changes needed for that kind of tuning.

## Data

Two accepted CSV shapes (see `load_keywords` above); every script must support both.
`data/keyword_real.csv` (521 real keywords, 12 niches via `source_niche`) and
`data/keywords_sample.csv` (47 synthetic keywords, precomputed-cpc format) are the
canonical fixtures — see `README.md` for the full per-niche breakdown before adding
new sample data.
