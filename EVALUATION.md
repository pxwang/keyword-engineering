# Evaluation

Validation results for this project's clustering, classification, and peer-finding
logic against real keyword data, plus the performance measurements behind the claims
made in the README. Everything below was re-run against the current code before being
written down here — none of it is carried over from earlier, possibly-stale runs.

Dataset used throughout unless noted otherwise: `data/keyword_real.csv`, 521 real
keywords with Google Ads search volume, CPC bid range, and competition data, accessed
via WordStream's Free Keyword Tool (WordStream is a UI layer over Google Ads Keyword
Planner data, not the origin of the data itself), across 12 known, unrelated niches
(`source_niche` column: home_repair, insurance, travel, hotel, mortgage, car_rental,
car_dealer, job_search, senior_care, wedding, shopping, pension).

## 1. Clustering: does it actually separate unrelated keywords?

At the default `--threshold 0.78`, clustering the full 521-keyword dataset (spanning
12 unrelated niches) produces **228 groups**. Checking every group against the known
`source_niche` ground truth: **1 group out of 228 mixes two niches.**

```
group  keyword               source_niche  cpc
194    engagement rings      shopping      $4.40
194    wedding rings         wedding       $2.65
194    mens wedding rings    wedding       $2.81
```

This isn't a clustering error — it's the model correctly recognizing that engagement
rings and wedding rings are the same jewelry sub-market. The niche labels themselves
(assigned when the source data was collected) drew an arguably-artificial boundary
here, not the clustering algorithm.

**Conclusion:** at `threshold=0.78`, cross-niche contamination is effectively zero
across a genuinely diverse 521-keyword, 12-niche dataset, with the one exception being
a defensible link rather than a mistake.

### Known failure mode: chaining

Connected-components clustering joins two keywords if their similarity is
`>= threshold`, transitively through any path of intermediate keywords — not only if
they're directly similar to each other. This produces occasional large, loosely-related
clusters. The clearest example on this dataset: a 41-member cluster anchored on
`"homeowners insurance quote"` includes `"auto insurance"`, which is only **0.53**
similar to that anchor directly — nowhere near the threshold. It got pulled in through
a real chain of direct `>=0.78` links:

```
auto insurance (0.886) → car insurance (0.810) → car insurance quotes (0.860)
  → insurance quotes (0.784) → home insurance quote (0.939) → homeowners insurance quote
```

Each arrow is a genuine direct link; the chain as a whole merges two different
insurance product lines (auto vs. home) that don't directly resemble each other.

Raising `--threshold` to **0.83** fixes this specific case cleanly. Re-running the
same dataset at 0.83: **300 groups** (vs. 228 at 0.78), and the insurance mega-cluster
splits into three separate, tight groups:

| Group | Members | Avg CPC |
|---|---|---|
| `car insurance quotes` | car insurance quotes, insurance quotes | $85.09 |
| `auto insurance` | auto insurance, car insurance | $88.22 |
| `homeowners insurance quote` | 6 home-insurance-quote phrasings | $100.49 |

Side effect worth knowing: raising the threshold to 0.83 also breaks the
engagement-rings/wedding-rings link from above (cross-niche groups become **0/300**
instead of 1/228) — that link was itself just above 0.78 and just below 0.83. Raising
the threshold trades recall (catching loosely-related phrasing, including that
legitimate link) for precision (avoiding chained mega-clusters).

**Conclusion:** `0.78` is a reasonable default that favors recall; `0.82`–`0.83` is the
right move whenever a cluster reads as "too broad" on inspection. There's no single
correct threshold — it depends on whether false-merges or false-splits are more costly
for the use case.

## 2. Classification: does classify-first predict the right category?

Running `keyword_classify.py` on the same 521 keywords and checking each keyword's
assigned `category` against its true `source_niche`:

| Niche | n | Top assigned category | Accuracy |
|---|---|---|---|
| job_search | 50 | Career & Employment | **94.0%** |
| car_dealer | 25 | Automotive -- For Sale | **92.0%** |
| travel | 36 | Travel | 75.0% |
| car_rental | 25 | Automotive -- For Sale | 76.0% |
| mortgage | 25 | Finance & Insurance | 80.0% |
| insurance | 97 | Finance & Insurance | 69.1% |
| shopping | 24 | Apparel / Fashion & Jewelry | 66.7% |
| home_repair | 124 | Home & Home Improvement | 63.7% |
| wedding | 25 | Personal Services (Weddings, Cleaners, etc.) | 60.0% |
| pension | 16 | **Unknown** | 62.5% |
| hotel | 49 | Travel | 51.0% |
| senior_care | 25 | Home & Home Improvement | 40.0% |

**Overall: 76 of 521 keywords (14.6%) were labeled `Unknown`** rather than forced into
a bad-fit category — this is the `unknown_threshold=0.35` safety net working as
designed, not a failure. Notably, **pension's largest bucket is now `Unknown`
(62.5%)** rather than a wrong confident guess — the taxonomy genuinely has no
"retirement benefits" category, and the classifier correctly reports low confidence
instead of picking a wrong one (in an earlier version without the Unknown fallback,
these were being force-assigned to "Government & Politics" and "Health & Fitness").

The two weakest *confident* results (senior_care 40%, hotel 51% as plurality, not
majority) reflect real gaps or ambiguity in the taxonomy, not classifier bugs:
- **senior_care** has no dedicated "elder care" category — its keywords split between
  Home & Home Improvement and Physicians & Surgeons, both defensible partial fits.
- **hotel** keywords split between Travel and Restaurants & Food — largely because the
  dataset is heavy with hotel *brand names* (Marriott, Hilton, Hyatt...) that carry
  little topical vocabulary either way; brand names are a known hard case for
  embedding-based classification in general (see Known limitations below).

**Enrichment mattered.** Before adding hint phrases (i.e. embedding just the bare
category label like `"Travel"`), the same evaluation showed "hotel" assigning to
"Restaurants & Food" over "Travel" as its single top pick, and "home_repair" splitting
across categories with only a 29% plurality (including 22 items nonsensically
assigned to "Dentists & Dental Services", purely on shared "near me / emergency /
affordable" service-ad phrasing). Adding 5 example phrases per category fixed the
hotel misassignment entirely and raised home_repair's accuracy from 29% to 63.7%.

**Conclusion:** classify-first is reliable for niches with distinct vocabulary
(job_search, car_dealer, travel) and honest about its limits where the taxonomy itself
has gaps (pension, senior_care) — reporting `Unknown` there is the correct behavior,
not a bug to fix.

## 3. Peer-finding: are the suggested "higher-value peers" actually relevant?

Running `keyword_peers.py` with defaults (`cpc-premium=0.3`, `min-sim=0.5`,
`cpc-weight=0.0`, `top-n=5`) on the same 521 keywords: **385 keywords (73.9%) have at
least one qualifying peer**, producing 1,523 (keyword, peer) rows. Every row was
verified to satisfy both constraints (`peer_cpc > cpc * 1.3` and `sim_score >= 0.5`)
with no exceptions.

The `min-sim` floor is doing real work. Without it, a keyword search for peers
degrades to "closest match among all higher-cpc keywords regardless of relevance" —
concretely, before the floor was added, `"plumber"` (which has exactly one genuine
peer, `"emergency plumber"`) was returning 4 additional irrelevant rows (assorted
insurance keywords at similarity **0.14–0.16**) just to pad out to `top-n=5`. With the
floor in place, `"plumber"` correctly returns only its one real peer instead of being
padded with noise.

**Conclusion:** the min-sim floor is necessary, not optional — `top-n` alone doesn't
prevent irrelevant results once a keyword's genuine peers run out.

## 4. Performance

### GPU vs. CPU (embedding step only)

Measured directly on this project's own `embed_keywords()` call path (Apple Silicon
MPS backend; CUDA would be expected to scale further given more parallel throughput
than MPS provides):

| Keywords | CPU | MPS (Apple GPU) | Speedup |
|---|---|---|---|
| 20,000 (synthetic) | 8.36s (2,394 kw/s) | 4.25s (4,701 kw/s) | ~1.97x |
| 50,000 (synthetic) | 20.49s (2,440 kw/s) | 10.92s (4,577 kw/s) | ~1.88x |
| 521 (real, this project's actual dataset) | ~6s (full pipeline, includes model load + clustering) | ~6s | negligible |

**Conclusion:** GPU only meaningfully helps the embedding step, and only once the
keyword count is large enough (roughly 10k+) that embedding time dominates over fixed
overhead (Python startup, model load). At this project's actual 521-keyword scale, GPU
vs. CPU is not worth choosing deliberately — the auto-detect default is fine either way.

### Clustering's O(n²) scaling (device-independent)

The clustering step (`cluster_by_threshold`) is pure Python/NumPy and does not benefit
from GPU at all. Measured scaling (confirms quadratic growth: ~4x time per doubling
of n, as expected for O(n²)):

| n | Time | Similarity matrix memory |
|---|---|---|
| 2,000 | 0.18s | 16 MB |
| 4,000 | 0.72s | 64 MB |
| 8,000 | 2.86s | 256 MB |
| 50,000 (extrapolated) | ~112s | ~10 GB |

The 50,000 row is extrapolated, not measured directly — on this machine's 17.2 GB of
total RAM, materializing a 10 GB dense similarity matrix was judged too risky to
attempt (would likely force swapping or an out-of-memory failure alongside whatever
else is running).

**Conclusion:** this is the real scaling ceiling in the current implementation, not
GPU/CPU choice. It's what motivates `--classify-first` (shrinks the problem to
per-category buckets) and the k-means size-cap fallback described as future work in
the README — neither of which is a GPU problem to solve.

### Embedding cache

Clean cold-vs-warm comparison, `--classify-first` on the full 521-keyword dataset:

| Run | Time | Output |
|---|---|---|
| Cold (empty cache) | 5.89s | |
| Warm (cache populated from the cold run) | 3.66s | |
| | **~38% reduction** | **byte-for-byte identical to the cold run** |

The warm run's `embed_keywords()` calls never instantiate the SentenceTransformer
model at all (every keyword and anchor/hint phrase is already cached), which is where
essentially all of the savings comes from.

**Conclusion:** the cache is a pure performance optimization — verified to never
change output — and its benefit is largest specifically for `--classify-first`, which
embeds the same keywords more than once per run (once for category assignment, again
per-bucket for clustering) without it.

## Known limitations (carried forward, still true)

- Brand names and acronyms often don't cluster with their expansions (e.g. "lv purse"
  vs. "louis vuitton purses") and are a known hard case for the classifier too (see
  hotel, above) — the model matches on lexical/semantic surface form, not real-world
  entity knowledge.
- Chaining (section 1) means cluster quality should be spot-checked, not assumed,
  especially for large or unusually broad-looking clusters.
- The 31-category taxonomy has real gaps (no dedicated elder-care or retirement
  category) that no amount of hint-phrase tuning fixes — that requires adding
  categories, not better hints.
