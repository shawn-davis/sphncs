"""Minimal sphncs usage example."""

from sphncs import SphncsClusterer


strings = [
    "order-001", "order-002", "order-003",
    "invoice-900", "invoice-901", "invoice-902",
]

model = SphncsClusterer(
    metric="normalized_levenshtein",
    clustering_mode="spectral_consensus",
    n_embeddings=3,
    consensus_n_clusters=2,
    random_state=7,
).fit(strings)

for string, label in zip(strings, model.labels_):
    print(f"{label}: {string}")
print("Representatives:", model.representatives_)
