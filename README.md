# sphncs

Licensed under [Apache-2.0](LICENSE).

`sphncs` (syntactical pseudo-hierarchical normal clustering for strings) clusters
strings by embedding an on-demand string distance with FastMap and finding density
separations with `KDEpy.FFTKDE`.

It supports a lightweight single-dimension mode and an optional spectral-consensus
mode that combines KDE cluster assignments from several FastMap dimensions.  An
optional first KDE can partition strings by length before syntactic clustering.
In spectral-consensus mode, automatic `k` is the median number of clusters
observed by the per-dimension KDE fits, rather than the number of embeddings.
Set `consensus_n_clusters` to `"min"`, `"mean"`, `"median"`, or `"max"` to
select another reduction of those observed counts; an explicit integer remains
available when a fixed target is required.
`extrema_prominence_fraction` controls how deep a KDE valley must be relative
to that KDE's density range (the default is `0.05`); smaller fractions preserve
more candidate modes.
The estimator uses three FastMap pivot-refinement passes and a safe distance cache
for its immutable string inputs by default; both are configurable with
`fastmap_iters` and `fastmap_distance_cache`.

```python
from sphncs import SphncsClusterer

model = SphncsClusterer(metric="normalized_levenshtein")
labels = model.fit_predict(["abc-001", "abc-002", "invoice-15", "invoice-16"])
print(model.representatives_)
```

### Optional log-field preprocessing

`log_filters` replaces selected fields with filter-specific, fixed-width masks
before embedding and clustering. Length partitioning uses the filtered strings
by default; pass `length_partitioning_before_filtering=True` to partition on
the original raw log lengths first. Every marker is four
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
model = SphncsClusterer(
    metric="char_ngram_jaccard",
    log_filters=["timestamp", "severity", "variable"],
)
```

### LogSPHNCS

`LogSPHNCS` is the log-specific entry point. It defaults to all log filters,
4-gram Jaccard, length partitioning, 10-dimensional spectral consensus, and
exact filtered-template compression. Raw records remain aligned in
`raw_strings_` and `raw_representatives_`.

```python
from sphncs import LogSPHNCS

model = LogSPHNCS().fit(log_lines)
```

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
