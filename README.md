# sphncs

Licensed under [Apache-2.0](LICENSE).

`sphncs` is the **Similarity-Preserving Hierarchical Nonparametric Clustering
System**. It clusters any objects with a well-defined, non-negative distance
metric by embedding on-demand distances with FastMap and finding density
separations with `KDEpy.FFTKDE`.

It supports a lightweight single-dimension mode and an optional spectral-consensus
mode that combines KDE cluster assignments from several FastMap dimensions. An
optional first KDE can partition objects with any user-supplied scalar feature.
In spectral-consensus mode, automatic `k` is the median number of clusters
observed by the per-dimension KDE fits, rather than the number of embeddings.
Set `consensus_n_clusters` to `"min"`, `"mean"`, `"median"`, or `"max"` to
select another reduction of those observed counts; an explicit integer remains
available when a fixed target is required.
`extrema_prominence_fraction` controls how deep a KDE valley must be relative
to that KDE's density range (the default is `0.05`); smaller fractions preserve
more candidate modes.
The estimator uses three FastMap pivot-refinement passes. Configure the number
of passes with `fastmap_iters`.

```python
from dataclasses import dataclass

from sphncs import SphncsClusterer


@dataclass
class Point:
    x: float


def distance(left: Point, right: Point) -> float:
    return abs(left.x - right.x)

model = SphncsClusterer(metric=distance)
labels = model.fit_predict([Point(0.0), Point(0.2), Point(10.0), Point(10.2)])
print(model.representatives_)
```

For optional first-stage partitioning, provide a scalar feature. Partitioning is
hard routing: objects in different feature intervals are clustered separately.

```python
model = SphncsClusterer(
    metric=distance,
    partitioning=True,
    partitioning_feature=lambda point: point.x,
)
```

String distances such as `normalized_levenshtein` and `char_ngram_jaccard`
remain available in `sphncs.distances`. They are conveniences, not a restriction
on the estimator's input type.

### Log clustering

`LogSPHNCS` is the log-specific adapter. Its `log_filters` replace selected
fields with filter-specific, fixed-width masks before embedding and clustering.
The optional initial partitioner uses filtered strings by default; pass
`partitioning_before_transform=True` to derive its feature from the original
raw strings first. It uses length by default;
set `partitioning_feature="entropy"` for character Shannon entropy or
`"normalized_entropy"` for character-use evenness. Every marker is four
characters long: timestamps use `<#T>`, severity uses `<#S>`, UUIDs use `<#U>`,
IPs use `<#I>`, hex values use `<#H>`, numbers use `<#N>`, paths use `<#P>`,
quoted values use `<#Q>`, and identifiers use `<#D>`. Choose individual filters
(`timestamp`, `severity`, `uuid`, `ip`, `hex`, `number`, `path`, `quoted`, and
`identifier`), use `variable` for all value-masking filters, or use `all` for
every filter.

The `hex` filter recognizes `0x`-prefixed values. Short bare hexadecimal
sequence fields such as `0000000e` are normalized by the `number` filter, so
they match their decimal-only counterparts without masking longer component IDs.

`representatives_` contains the filtered form by default. The original training
lines remain available in the position-aligned `raw_strings_`, through
`get_raw_string(index)`, and as `raw_representatives_` for the cluster
representatives.

```python
model = LogSPHNCS(
    log_filters=["timestamp", "severity", "variable"],
)
```

### LogSPHNCS

`LogSPHNCS` defaults to all log filters,
4-gram Jaccard, length partitioning, 10-dimensional spectral consensus, and
exact filtered-template compression. Raw records remain aligned in
`raw_strings_` and `raw_representatives_`.

```python
from sphncs import LogSPHNCS

model = LogSPHNCS().fit(log_lines)
```

### Model persistence

Fitted models can be saved to a versioned, integrity-checked `.sphncs` archive
and loaded later. The archive preserves prediction and transformation state.
Each FastMap projection is persisted with FastMapy’s native versioned format.

```python
model.save("logs.sphncs")
restored = LogSPHNCS.load("logs.sphncs")
assert restored.predict(log_lines).tolist() == model.predict(log_lines).tolist()
```

Archives use Python pickle to support arbitrary input objects and user-supplied
metrics or transformers. Only load archives from sources you trust. Callables
must be importable functions (rather than lambdas or nested functions) to save
reliably. Format version 1 is validated on load; unsupported future formats are
rejected rather than loaded incorrectly.

`fastmapy` is included as the `vendor/fastmapy` Git submodule and isolated behind
an embedding adapter. It is used for all FastMap fits, including its `fit_many`
API for spectral-consensus embeddings; no pairwise string-distance matrix is
created or retained.

## Development

```bash
git submodule update --init --recursive
python -m pip install -e vendor/fastmapy
python -m pip install -e '.[dev]'
python -m pytest
```
