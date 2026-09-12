# sphncs

Licensed under [Apache-2.0](LICENSE).

`sphncs` (syntactical pseudo-hierarchical normal clustering for strings) clusters
strings by embedding an on-demand string distance with FastMap and finding density
separations with `KDEpy.FFTKDE`.

It supports a lightweight single-dimension mode and an optional spectral-consensus
mode that combines KDE cluster assignments from several FastMap dimensions.  An
optional first KDE can partition strings by length before syntactic clustering.

```python
from sphncs import SphncsClusterer

model = SphncsClusterer(metric="normalized_levenshtein")
labels = model.fit_predict(["abc-001", "abc-002", "invoice-15", "invoice-16"])
print(model.representatives_)
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
