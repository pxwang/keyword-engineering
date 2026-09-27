# Keyword Engineering

[![Tests](https://github.com/pxwang/keyword-engineering/actions/workflows/tests.yml/badge.svg)](https://github.com/pxwang/keyword-engineering/actions/workflows/tests.yml)

Three tools that share the same embedding pipeline (all-MiniLM-L6-v2), working on the
same keyword+CPC data:

1. **Classification** (`keyword_classify.py`) — bucket keywords into coarse industry
   categories, so later steps can work per-category instead of over everything at once.
2. **Clustering** (`keyword_cluster.py`) — group near-duplicate keyword phrasings
   together and rank by `similarity_to_representative * cpc`, optionally per-category.
3. **Peer finding** (`keyword_peers.py`) — for each keyword, surface other keywords
   that are similar but meaningfully higher-value, as candidates worth considering.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Every command below assumes the venv is active (`source .venv/bin/activate`). To leave
it: `deactivate`.

GPU: every script auto-detects the best available device — CUDA, then Apple Silicon
MPS, then CPU — no setup needed. Override with `--device cpu|mps|cuda` if you want to
force one. At the ~500-keyword scale of the datasets here, device choice doesn't
meaningfully affect runtime; it starts mattering around 10k+ keywords (see *Further
work* below).

## Testing

```bash
pytest
```

Unit tests for the deterministic logic in all four modules — `load_keywords`'s cpc
interpolation, `cluster_by_threshold`'s union-find grouping, `rank_terms`'s
group-averaging and sort order, `find_peers`'s cpc-premium/min-sim/top-n filtering,
`classify`'s nearest-anchor assignment and Unknown fallback, and the embedding cache's
hit/miss behavior. These run in a few seconds, need no model download, and use
hand-crafted fixed vectors or a fake model (monkeypatched in) instead of the real
MiniLM model, so they're fast and fully deterministic.

They test *code correctness* — does the logic do what it's supposed to on known
inputs — not *result quality* on real keyword data. For that, see
[`EVALUATION.md`](EVALUATION.md): validation against this project's own real,
labeled keyword data (does clustering actually separate unrelated niches, does
classify-first actually predict the right category, etc.), plus the GPU/CPU and
caching benchmarks.

## Embedding cache

All three tools call the same `embed_keywords()` helper (in `keyword_common.py`),
which transparently caches every embedding it computes in a local SQLite database at
`cache/embeddings.db`, keyed by `(model_name, keyword)`. A keyword is only ever sent
through the model once — any later call, whether from a different script, a different
run, or a different bucket within the same `--classify-first` run, gets served from
the cache instead of recomputing.

This isn't just a cross-run convenience: `--classify-first` used to embed every
keyword twice in a single run (once in `keyword_classify.py`'s category assignment,
again per-bucket in `keyword_cluster.py`'s clustering step) before the cache existed.
Measured on the 521-keyword real dataset: **~38% faster** warm vs. cold, with
byte-for-byte identical output either way, since caching only skips redundant
computation and never changes what gets computed — full numbers in
[`EVALUATION.md`](EVALUATION.md#embedding-cache).

The cache is pure local infrastructure, not project data — `cache/` is gitignored. To
bypass it entirely (e.g. to benchmark raw embedding speed), pass `cache_path=None` if
calling `embed_keywords()` directly; none of the CLI scripts expose a flag for this
today since there's no real reason to disable it in normal use.

## Input format

Two accepted shapes, both need a `keyword` column, and both work with all three tools:

**Raw bid range** (what `data/keyword_real.csv` uses — reconstructed straight from the
Google Ads "top of page bid" low/high columns shown by WordStream's Free Keyword Tool,
which is a UI over Google Ads Keyword Planner data, not the data's origin):

```csv
keyword,cpc_low,cpc_high,search_volume,competition,source_niche
wedding rings,1.10,4.20,246000,High,wedding
```

`source_niche` records which original scrape batch a row came from (see *Datasets*
below); it's carried through to output but not used in any math. `cpc` is derived at
run time as `cpc_low + (cpc_high - cpc_low) * cpc_weight` (flag name varies slightly
per tool — see each tool's options table), so the same input file can be re-scored
under different CPC assumptions without re-scraping anything:

```bash
python keyword_cluster.py --input data/keyword_real.csv --cpc-weight 0.3   # conservative, near the low bid
python keyword_cluster.py --input data/keyword_real.csv --cpc-weight 1.0   # aggressive, the high bid
```

**Precomputed CPC** (used by `data/keywords_sample.csv`, a synthetic demo set):

```csv
keyword,cpc
running shoes,1.85
```

Here `cpc` is used as-is and the cpc-weight flag is ignored.

## Datasets in `data/`

| File | Keywords | Contents |
|---|---|---|
| `keyword_real.csv` | 521 | Real Google Ads keyword data (accessed via WordStream's Free Keyword Tool), spanning 12 unrelated niches (see `source_niche` column), concatenated into one file |
| `keywords_sample.csv` | 47 | Synthetic demo set (running shoes, coffee, laptops, yoga, protein) — uses the simpler precomputed `cpc` format instead of `cpc_low`/`cpc_high` |

`source_niche` values in `keyword_real.csv`:

| Niche | Keywords |
|---|---|
| home_repair | 124 (home repair, handyman, plumber, electrician, roofing) |
| insurance | 97 (insurance, life insurance, home insurance, annuities) |
| job_search | 50 (general + engineering) |
| hotel | 49 (hotels, vacation rentals) |
| travel | 36 (flights, air travel) |
| car_dealer | 25 |
| car_rental | 25 |
| mortgage | 25 |
| senior_care | 25 |
| wedding | 25 |
| shopping | 24 |
| pension | 16 |

---

## 1. Classification

`keyword_classify.py` assigns each keyword to whichever of a small set of coarse
industry categories it's most similar to, by cosine similarity over the same
embeddings used everywhere else in this project. It exists to make clustering scale
(see below) and as a standalone way to sanity-check how a keyword set breaks down.

```bash
python keyword_classify.py --input data/keyword_real.csv
```

The category list, and a handful of example "hint" phrases per category used to build
a richer anchor embedding than the bare category name alone, live in
`config/categories.yaml` — edit that file to add categories, add hints, or tune the
unknown threshold, no code changes needed. The default 31 category names started from
a real ad-industry taxonomy (transcribed by hand, not fetched live), but the hint
phrases were hand-written for this project, not sourced from anywhere.

A bare category label alone doesn't embed well against real keyword phrasing (e.g.
"hotel" scored higher on "Restaurants & Food" than on the single word "Travel" using
bare labels) — the hint phrases fix this by giving each category real keyword-style
vocabulary to match against.

If even the best-matching category is a weak match, the keyword is labeled `"Unknown"`
instead of being forced into a bad fit — controlled by `unknown_threshold` in the YAML
(default `0.35`, picked from this project's own score distribution: it flags roughly
the bottom 15% of weakest matches). The output CSV keeps both `category` (possibly
`"Unknown"`) and `closest_category` (the best match regardless of confidence), so you
can always see what it would have picked.

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `data/keywords_sample.csv` | Input CSV |
| `--output` | `output/keyword_categories.csv` | Classified CSV |
| `--report` | `output/keyword_categories_report.md` | Human-readable grouped report (`''` to skip) |
| `--categories` | `config/categories.yaml` | YAML file defining categories and hint phrases |
| `--unknown-threshold` | (from YAML, `0.35`) | Override the YAML file's threshold |
| `--cpc-range-weight` | `0.5` | Same meaning as `keyword_cluster.py`'s `--cpc-weight` |

Accuracy against this project's own 12 known niches, per-niche breakdown, and what the
`Unknown` fallback actually catches: see [`EVALUATION.md`](EVALUATION.md#2-classification-does-classify-first-predict-the-right-category).

## 2. Clustering

```bash
python keyword_cluster.py --input data/keyword_real.csv --threshold 0.78
```

Groups near-duplicate keyword phrasings together (connected-components clustering:
join two keywords whenever their cosine similarity is ≥ `--threshold`, transitively —
see *Known limitations* below), then ranks by `similarity_to_representative * cpc`.

This prints a per-group breakdown to stderr and writes two files: the full ranked table
(`output/keyword_groups.csv`) and a grouped, human-readable markdown report
(`output/keyword_report.md`) — clusters as sections sorted by average cluster value,
members listed underneath their representative term. The CSV is for further
processing; the report is for actually reading through the clusters (`Cmd+F` a
keyword, scroll through sections, no need to sort/filter columns by hand).

By default the report includes every group, including singletons (terms that didn't
cluster with anything). To see only real multi-term clusters:

```bash
python keyword_cluster.py --input data/keyword_real.csv --report-min-size 2
```

### Classify-first, for scale

Clustering is O(n²) — fine at hundreds of keywords, a real problem at tens of
thousands (see *Further work*). `--classify-first` runs the classifier above first,
then clusters independently within each resulting category, then stitches everything
back into one globally-ranked CSV/report:

```bash
python keyword_cluster.py --input data/keyword_real.csv --classify-first
```

On the 521-keyword real dataset: 228 groups without `--classify-first` vs. 267 with
it — more, smaller groups, since cross-category chaining is no longer possible by
construction (verified: zero groups span more than one category).

**Important:** `--classify-first` only prevents merges *across* categories — it does
**not** fix chaining *within* one. The 41-member "insurance quotes" mega-cluster (see
*Known limitations*) forms identically whether or not `--classify-first` is on,
because every one of those keywords lands in the same "Finance & Insurance" bucket.
Raising `--threshold` is still the right lever for that; `--classify-first` solves a
different problem (scale + category boundaries), not clustering quality within a
category.

### Output columns

- `group` — cluster id
- `representative_term` — the highest-CPC keyword in that group
- `similarity_to_rep` — cosine similarity of this keyword to the representative
- `cpc` — derived CPC
- `score` — `similarity_to_rep * cpc`, the per-keyword ranking value
- `group_size` — how many keywords support this cluster
- `group_cpc_avg` — average CPC across the cluster's members, used to rank clusters
  against each other (average, not sum — members are near-duplicate phrasings of the
  same intent, not independent revenue streams, so summing would just reward clusters
  for having more members)

Rows are sorted by `group_cpc_avg`, then by `score` within each cluster.

### CLI options

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `data/keywords_sample.csv` | Input CSV |
| `--output` | `output/keyword_groups.csv` | Where to write the full ranked CSV |
| `--report` | `output/keyword_report.md` | Where to write the human-readable grouped report (`''` to skip) |
| `--report-min-size` | `1` | Omit groups smaller than this from the report; `2` hides singletons |
| `--threshold` | `0.78` | Cosine similarity cutoff for grouping terms into the same cluster |
| `--cpc-weight` | `0.5` | Where to sample `cpc` within `[cpc_low, cpc_high]` — `0.0`=low bid, `1.0`=high bid, `0.5`=midpoint |
| `--classify-first` | off | Classify into categories first, cluster independently per category |
| `--categories` | `config/categories.yaml` | Category/hints YAML (only used with `--classify-first`) |
| `--unknown-threshold` | (from YAML) | Override the YAML unknown threshold (only used with `--classify-first`) |
| `--model` | `all-MiniLM-L6-v2` | Any sentence-transformers model name |
| `--device` | auto-detect | `cuda`, `mps`, or `cpu` |

### Known limitations

- Connected-components clustering can **chain**: if A is similar to B and B is similar
  to C, all three land in one group even if A and C aren't directly similar. Raising
  `--threshold` (try `0.82`–`0.83`) tightens this.
- Brand names and acronyms often don't cluster with their expansions (e.g. "lv purse"
  vs "louis vuitton purses") — the model clusters on semantic/lexical surface form,
  not real-world domain knowledge.

Concrete numbers, the specific chaining example on real data, and the 0.78-vs-0.83
threshold comparison: [`EVALUATION.md`](EVALUATION.md#1-clustering-does-it-actually-separate-unrelated-keywords).

## 3. Keyword peers (finding higher-value alternatives)

`keyword_peers.py` is a separate tool from clustering: for each keyword, it finds
*other* keywords that are (a) semantically similar and (b) meaningfully higher CPC —
"if I'm targeting X, what similar-but-more-valuable keyword Y should I also consider?"

```bash
python keyword_peers.py --input data/keyword_real.csv
```

A candidate `keyword2` counts as a peer of `keyword1` only if both hold:
- `keyword2.cpc > keyword1.cpc * (1 + cpc_premium)` — meaningfully higher value (default: 30% higher)
- `similarity(keyword1, keyword2) >= min_sim` — meaningfully related (default: 0.5)

Peers are ranked by `score = peer_cpc * cpc_weight + sim_score`. The default
`cpc_weight=0` ranks purely by similarity among the qualifying candidates — the peer
list surfaces "closest match," not "most expensive match." Raise `cpc_weight` to start
favoring higher-CPC peers over closer-but-cheaper ones.

Both output files are ordered by each keyword's best (rank-1) peer score, descending —
strongest matches lead the file, weaker/noisier ones trail toward the bottom.

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `data/keywords_sample.csv` | Input CSV |
| `--output` | `output/keyword_peers.csv` | Ranked CSV, one row per (keyword, peer) pair, sorted by best-peer score |
| `--report` | `output/keyword_peers_report.md` | Human-readable grouped report (`''` to skip) |
| `--cpc-premium` | `0.3` | Required cpc uplift to qualify as a peer (30%) |
| `--min-sim` | `0.5` | Required similarity to qualify as a peer at all |
| `--cpc-weight` | `0.0` | Weight on `peer_cpc` in the ranking score |
| `--top-n` | `5` | Max peers kept per keyword |
| `--cpc-range-weight` | `0.5` | Same as `keyword_cluster.py`'s `--cpc-weight`, renamed to avoid confusion with the peer-ranking `--cpc-weight` above |

Example output (defaults — the actual top of the sorted output):

```
the handyman                       -> handyman                       (cpc=$7.05,  +65%, sim=0.96)
average price of roof replacement  -> roof replacement cost          (cpc=$13.37, +48%, sim=0.96)
site for searching jobs            -> site to find jobs              (cpc=$2.19,  +56%, sim=0.93)
thrift stores near me              -> thrift store                   (cpc=$2.75,  +38%, sim=0.93)
senior care service                -> senior citizen care services   (cpc=$5.71,  +60%, sim=0.93)
...
auto insurance                     -> (no peers — already near the top of the cpc range)
```

Without the `--min-sim` floor, keywords with few genuine peers get padded with
irrelevant high-CPC filler just because it's the "least dissimilar" thing left in the
candidate pool — that's why the floor defaults on; see
[`EVALUATION.md`](EVALUATION.md#3-peer-finding-are-the-suggested-higher-value-peers-actually-relevant)
for the concrete before/after. Note: `keyword_peers.py` currently has no
`--classify-first` mode — it searches the whole input file regardless of category,
which is fine at this dataset's scale (see *Further work*).

---

## Further work / ideas for an end-to-end picture

This project currently covers classify → cluster → find-peers as three independently
runnable steps. A few things came up as natural next steps that aren't built:

- **K-means size-cap fallback.** `--classify-first` shrinks the clustering problem to
  per-category buckets, but a single category could still be too large for exact
  O(n²) clustering at real production scale (tens of thousands of keywords in one
  industry). The plan discussed: for any bucket still above a size cap after
  classification, split it further with a cheap approximate method
  (`sklearn.cluster.MiniBatchKMeans`, not full k-means — it doesn't hold all the data
  in memory for full Lloyd's-iteration passes) before running the exact clusterer on
  each piece. Simplest to implement as a flat `while` loop over a worklist of buckets
  (push oversized buckets back on with their k-means splits) rather than a recursive
  function — same logic, easier to read, no call-stack depth to reason about.
- **Wiring `--classify-first` into `keyword_peers.py`.** Peer-finding currently
  searches the whole dataset; restricting candidates to the same category would give
  it the same scaling benefit clustering already has. Deliberately left out for now —
  cross-category peer suggestions might sometimes be desirable (e.g. surfacing a
  higher-value keyword in an adjacent category you hadn't considered), so it's not a
  strict win the way it is for clustering.
- **A "replace low-value keyword with higher-value peer" step.** The natural next
  action after peer-finding: consume `keyword_peers.csv` and produce a "recommended
  swaps" list. Not built, and worth being deliberate about before automating it —
  **higher CPC isn't the same as "better keyword to target."** CPC reflects auction
  competition, not conversion rate or actual return. A cheaper keyword with healthy
  volume can easily out-earn a pricier "peer" that's merely semantically similar.
  Peers should be treated as candidates for human review or A/B testing unless real
  conversion/revenue data per keyword is available to validate the swap against —
  similarity + CPC alone isn't a sound basis for automatic replacement.
- **Production scale: distributed bucket-then-cluster.** Classify-first buckets are
  embarrassingly parallel — each can be clustered independently on a different
  machine/worker with no shared state, since by construction no cluster ever spans
  two buckets. GPU meaningfully speeds up the embedding step at scale but does *not*
  speed up the clustering step itself, which is pure Python/NumPy and hits a real O(n²)
  time and memory ceiling at large single-bucket sizes — measured numbers in
  [`EVALUATION.md`](EVALUATION.md#4-performance) — which is exactly what the k-means
  fallback above is meant to prevent.
- **Category taxonomy quality.** The current 31 categories and their hint phrases are
  a reasonable starting point, hand-tuned against this project's own 12 niches, but
  two real gaps showed up (no dedicated "elder care" or "retirement benefits"
  category) and the hints themselves were hand-written, not sourced from real keyword
  data. A more rigorous version could derive hint phrases from actual labeled keyword
  exemplars per category instead of hand-written examples, or swap the whole
  nearest-anchor approach for a proper zero-shot NLI classifier.
