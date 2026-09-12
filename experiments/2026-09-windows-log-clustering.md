# Windows log-clustering experiment

## Scope

This experiment used the first 1,000 records in `Windows.log` from the
[LogPai Windows corpus](https://zenodo.org/record/3227177/files/Windows.tar.gz).
The target configuration is character 4-gram multiset Jaccard distance,
10 FastMap dimensions, spectral consensus, and length partitioning. Metrics
are measured in the processed-string space unless stated otherwise.

The final input uses all log filters and has 338 distinct processed templates.
Raw records remain position-aligned for traceability; they are not used to
measure cluster fitness after preprocessing.

## Changes made

### Spectral SPHNCS

- Added selectable relative KDE valley prominence
  (`extrema_prominence_fraction`). A 1% threshold preserves small modes and
  gives the best high-resolution baseline in this corpus.
- Consensus `k` now derives from the per-embedding observed KDE cluster counts.
  `min`, `mean`, `median`/`auto`, and `max` are supported rather than treating
  the number of embeddings as `k`.
- Identical per-embedding label signatures are coalesced after spectral
  clustering, and ISJ falls back to Silverman only when automatic ISJ cannot
  fit a small/discrete partition.

### FastMap and metric work

- Updated the `fastmapy` submodule to the cache-enabled revision and pass
  pivot-refinement and distance-cache options through the embedding adapter.
- Cached character n-gram counters for immutable string inputs.
- Added optional exact input-template compression, particularly valuable after
  log-field preprocessing.

### Log API and preprocessing

- Added `LogSPHNCS` as the explicit log-focused entry point, with 4-gram
  Jaccard, length partitioning, 10 dimensions, all filters, and template
  compression as defaults.
- Added `LogPreprocessor` with fixed-width, type-specific masks: timestamp,
  severity, UUID, IP, hex, number, path, quoted value, and identifier.
- Representatives default to processed strings; `raw_strings_`,
  `raw_representatives_`, and `get_raw_string()` preserve source-log access.
- Added an option to partition on raw lengths before preprocessing. Filtered
  lengths remain the default.
- Narrowed hexadecimal handling after diagnosing CSI records: `0x...` uses the
  hex marker; 7--8 character bare hexadecimal sequence fields are normalized
  as numbers; longer component IDs remain intact.

### Optional refinement prototype

`examples/refine_sphncs_with_crp.py` implements a blocked, SPHNCS-seeded
Dirichlet-process refinement on unique processed templates. It supports
cohesion gating and local or global medoid Davies--Bouldin (DB) move gates.
The global gate is deliberately an experimental script rather than part of the
public estimator API.

## Runtime results

Times are end-to-end `fit_seconds` on the same 1,000 records. The baseline is
the original raw-log configuration. The optimized/raw column isolates most
FastMap and metric improvements; the final preprocessed column additionally
benefits from preprocessing and duplicate-template compression, so it is not a
single-variable speed comparison.

| Dimensions | Original raw (s) | Optimized raw (s) | Final preprocessed (s) |
|---:|---:|---:|---:|
| 2 | 10.63 | — | 0.97 |
| 5 | 34.25 | 7.06 | 3.04 |
| 10 | 77.68 | 11.12 | 7.40 |
| 15 | 126.55 | 16.75 | 9.75 |
| 20 | 184.27 | 21.57 | 12.54 |

The observed original scaling is roughly linear in embedding count, matching
FastMap's repeated one-dimensional embedding work. Caching and template
compression remove repeated string-distance/shingle work that dominated this
small corpus.

## Final-method comparison

All entries below use the corrected, narrow bare-hex preprocessing input and
the same 4-gram Jaccard evaluation matrix. Lower DB and scatter are better;
higher silhouette and `<= 0.25` coverage are better. Parser run times were
captured separately in their benchmark artifact and are not inferred from this
table.

| Method | Clusters | Singletons | Silhouette | Medoid DB | Mean scatter | P95 distance | <= 0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LogSPHNCS, 10D, 1% prominence, max consensus | 39 | 9 | 0.9201 | 0.6263 | 0.05424 | 0.10870 | 0.978 |
| Drain3 | 37 | 11 | 0.9005 | 0.4869 | 0.02956 | 0.10870 | 0.971 |
| Spell | 26 | 6 | 0.8258 | 0.9696 | 0.09609 | 0.42748 | 0.848 |
| IPLoM | 21 | 3 | 0.8472 | 0.7737 | 0.11134 | 0.41667 | 0.940 |
| LogCluster | 37 | 14 | 0.9110 | 0.6820 | 0.02789 | 0.10870 | 0.973 |

The baseline LogSPHNCS configuration has the strongest silhouette and threshold
coverage of the parser baselines, while Drain3 retains the better DB and mean
scatter scores.

## CRP refinement results

The refinement starts from the final LogSPHNCS labels, uses alpha 1, total
Dirichlet prior mass 10, twelve sweeps, and the two nearest active medoids as
reassignment candidates. It operates on processed-template blocks, weighted by
their source-record multiplicity.

| Refinement | Clusters | Singletons | Silhouette | Medoid DB | Mean scatter | <= 0.25 | Refinement (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed LogSPHNCS | 39 | 9 | 0.9201 | 0.6263 | 0.05424 | 0.978 | — |
| Cohesion gate | 39 | 8 | **0.9333** | 0.5345 | 0.04297 | **0.981** | 0.72 |
| Local DB gate | 39 | 9 | 0.9274 | 0.5224 | 0.03839 | 0.980 | 0.69 |
| Global DB gate | 40 | 9 | 0.9298 | **0.4336** | **0.03918** | **0.981** | 0.98 |

The global DB gate evaluates the full weighted medoid-DB objective for each
proposed move. It rejected 38 additional DB-worsening moves and reduced DB by
31% relative to the seed, at a 0.0036 silhouette reduction versus
cohesion-only refinement.

An explicit greedy table-merge experiment was tried and then removed from the
implementation. It drove DB to 0.1603 by collapsing 40 clusters to 31, but
silhouette fell to 0.8720 and `<= 0.25` coverage to 0.956. That result shows
why DB alone is unsuitable as a whole-cluster merge objective.

## Reproduction artifacts

- `examples/regenerate_filtered_benchmark_input.py` rebuilds the raw/processed
  benchmark input.
- `examples/compare_log_cluster_fitness.py` runs LogSPHNCS and parser fitness
  comparisons.
- `examples/benchmark_template_parsers.py` runs Drain3, Spell, IPLoM, and
  LogCluster.
- `examples/refine_sphncs_with_crp.py` runs the optional blocked refinement.
- `outputs/` contains the JSON metrics, diagnostic data, KDE figure, and UMAP
  visualization used in this report.

Validation at the checkpoint: `pytest -q` reports 23 passing tests.
