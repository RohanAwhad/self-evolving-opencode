# Embedding-Based Text Clustering (No LLM) — Deep Research Report

> Generated: 2026-05-20 | Sources: 17 web + 0 code

## TL;DR

- **HDBSCAN is the wrong tool for <200 items** — its creator (Leland McInnes) explicitly says density estimation fails at small sample sizes ([GitHub #200](https://github.com/scikit-learn-contrib/hdbscan/issues/200))
- **Agglomerative clustering (average linkage + cosine distance)** is the best default for 50-200 short texts — deterministic, no noise labels, dendrogram inspection
- **Mean subtraction is the highest-impact preprocessing** — subtract corpus mean embedding to remove shared domain signal, exposing sub-topic variation. Essentially free.
- **Skip UMAP for <100 items** — manifold learning needs enough neighbors; at this scale it produces artifacts
- **Embedding model matters more than algorithm** — `all-mpnet-base-v2` beats `all-MiniLM-L6-v2` by ~1.3 V-measure on MTEB clustering benchmarks

## Overview

When clustering ~50-200 short text strings (user goals, intents, task descriptions) using sentence-transformer embeddings, the standard UMAP+HDBSCAN pipeline commonly recommended for topic modeling breaks down. HDBSCAN is a density-based algorithm that estimates the probability density function of the data — with <200 points, there isn't enough data to learn the density structure reliably.

The core problem for narrow-domain text (e.g., all software engineering goals) is **anisotropy**: sentence-transformer embeddings cluster in a narrow cone in embedding space, so pairwise cosine similarities are 0.85-0.99. Standard clustering sees one dense blob. Mean subtraction removes this shared domain signal, exposing the residual variation that distinguishes sub-topics.

## Key Findings

### 1. Algorithm Recommendation: Agglomerative Clustering

**Use `scipy.cluster.hierarchy.linkage` with `method='average'` on cosine distances.**

Why agglomerative wins for this use case:
- No noise labels (every point gets a cluster) — unlike HDBSCAN which labels 25-60% as noise on small datasets ([GitHub #72](https://github.com/scikit-learn-contrib/hdbscan/issues/72))
- Dendrograms allow visual inspection and natural cut-point selection
- Deterministic — same input always produces same output
- Works directly on precomputed cosine distance matrices
- O(n²) memory, O(n³) time — irrelevant for <200 items

**Linkage method**: Use `average` (UPGMA). `Ward` requires Euclidean distance. `Single` suffers from chaining. `Complete` produces tight but sometimes too-small clusters. ([sklearn docs](https://scikit-learn.org/stable/modules/clustering.html))

**Choosing number of clusters automatically**:
- Distance threshold: `scipy.cluster.hierarchy.fcluster(Z, t=threshold, criterion='distance')`
- Silhouette score optimization: try K=2..sqrt(n), pick max silhouette
- Dendrogram gap: find the largest vertical gap in the dendrogram — the natural cut point

**Enforcing max_cluster_size**: Post-hoc bisection — recursively split oversized clusters using the same linkage sub-tree or K-Means bisection.

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances

# 1. Compute condensed distance matrix
dist_matrix = cosine_distances(embeddings)
condensed = squareform(dist_matrix)

# 2. Linkage
Z = linkage(condensed, method='average')

# 3. Cut at distance threshold (or optimize silhouette)
labels = fcluster(Z, t=0.5, criterion='distance')
```

### 2. Preprocessing: Mean Subtraction (Highest Impact)

**Subtract the corpus mean embedding before clustering.** This removes the shared domain signal that makes all software engineering goals look similar.

The intuition: if every goal mentions "code", "MR", "fix", etc., the mean embedding captures this shared signal. Subtracting it leaves only the *differences* between goals — which is exactly what clustering needs.

```python
mean = embeddings.mean(axis=0)
centered = embeddings - mean
# Then L2-normalize
norms = np.linalg.norm(centered, axis=1, keepdims=True)
norms[norms == 0] = 1
normalized = centered / norms
```

**All-But-The-Top (ABTT)**: After centering, remove the top 1-3 principal components (which capture domain-level variance, not sub-topic variance). Mu & Viswanath 2018 ([arXiv:1702.01417](https://arxiv.org/abs/1702.01417)):

```python
from sklearn.decomposition import PCA

centered = embeddings - embeddings.mean(axis=0)
pca = PCA(n_components=3)
pca.fit(centered)
# Remove top 3 principal components
components = pca.components_  # (3, dim)
projection = centered @ components.T @ components  # (n, dim)
cleaned = centered - projection
# L2-normalize
cleaned = cleaned / np.linalg.norm(cleaned, axis=1, keepdims=True)
```

### 3. Alternative: Community Detection (Leiden + CPM)

For graph-based clustering with native max_cluster_size support:

1. Build a k-NN graph from cosine similarities
2. Run Leiden algorithm with CPM (Constant Potts Model) — resolution parameter controls granularity
3. Leiden `leidenalg` package supports `max_comm_size` natively

This avoids the threshold selection problem of sentence-transformers' `util.community_detection()` (which is actually just greedy threshold-based clustering, not a real community detection algorithm).

```python
import igraph as ig
import leidenalg

# Build k-NN graph from cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
# Keep only edges above threshold (or top-k neighbors)
threshold = np.percentile(sim_matrix[sim_matrix < 1.0], 75)
adj = (sim_matrix > threshold).astype(float)
np.fill_diagonal(adj, 0)

G = ig.Graph.Weighted_Adjacency(adj.tolist(), mode='undirected')
partition = leidenalg.find_partition(
    G, leidenalg.CPMVertexPartition,
    resolution_parameter=0.1,
    max_comm_size=10,  # native max cluster size!
)
labels = partition.membership
```

### 4. Embedding Model Choice

| Model | MTEB Clustering | Params | Speed |
|-------|----------------|--------|-------|
| all-MiniLM-L6-v2 | 42.35 | 22M | 5x faster |
| all-mpnet-base-v2 | 43.69 | 110M | baseline |
| BGE-large | ~48+ | 335M | slower |

For 50-200 items (one-shot embedding), use `all-mpnet-base-v2` — embedding cost is negligible. ([MTEB paper](https://ar5iv.labs.arxiv.org/html/2210.07316))

### 5. What NOT to Do

- **Don't use UMAP for <100 items** — `n_neighbors` becomes a significant fraction of dataset size, producing artifacts ([BERTopic docs](https://maartengr.github.io/BERTopic/getting_started/dim_reduction/dim_reduction.html))
- **Don't use HDBSCAN for <200 items** — density estimation fails ([McInnes, GitHub #200](https://github.com/scikit-learn-contrib/hdbscan/issues/200))
- **Don't use Ward linkage with cosine distance** — Ward requires Euclidean
- **Don't skip mean subtraction for narrow-domain text** — it's the single highest-impact preprocessing step

## Practical Guide

### Recommended pipeline for 50-200 short texts:

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 1. Embed
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(goals, convert_to_numpy=True)

# 2. Mean subtraction (critical for narrow-domain)
centered = embeddings - embeddings.mean(axis=0)

# 3. L2-normalize
norms = np.linalg.norm(centered, axis=1, keepdims=True)
norms[norms == 0] = 1
normalized = centered / norms

# 4. Cosine distance matrix → condensed form
dist = cosine_distances(normalized)
condensed = squareform(dist, checks=False)

# 5. Agglomerative clustering (average linkage)
Z = linkage(condensed, method='average')

# 6. Cut — either fixed threshold or optimize silhouette
labels = fcluster(Z, t=0.8, criterion='distance')

# 7. Post-hoc: bisect any cluster > max_size
# (recursive K-Means or re-cut the sub-dendrogram)
```

### Parameter tuning:
- **Distance threshold `t`**: Start at 0.5-0.8. Lower = more clusters. Inspect dendrogram to find natural gaps.
- **Silhouette optimization**: Try `t` from 0.3 to 1.0 in steps of 0.05, pick max average silhouette.

## Gotchas & Pitfalls

1. **cosine_distances returns float32 from float32 inputs** — HDBSCAN precomputed needs float64. Always `.astype(np.float64)`.
2. **scipy linkage expects condensed distance matrix** — use `squareform()` to convert NxN → condensed. Passing the full NxN matrix will silently produce wrong results.
3. **Ward linkage + cosine = wrong** — Ward minimizes variance, which is only valid for Euclidean distance.
4. **Mean subtraction order matters** — do it BEFORE normalization, not after. Centering + renormalizing is the correct sequence.
5. **fcluster labels start at 1, not 0** — subtract 1 if you want 0-indexed labels.
6. **`all-MiniLM-L6-v2` already partially normalizes** embeddings but NOT perfectly — always explicitly L2-normalize.

## Sources

1. McInnes (HDBSCAN creator) on small datasets — https://github.com/scikit-learn-contrib/hdbscan/issues/200
2. HDBSCAN excessive noise — https://github.com/scikit-learn-contrib/hdbscan/issues/72
3. MTEB Benchmark (Muennighoff et al., EACL 2023) — https://ar5iv.labs.arxiv.org/html/2210.07316
4. Petukhova et al. 2024, Text Clustering with LLM Embeddings — https://arxiv.org/html/2403.15112v5
5. Mu & Viswanath 2018, All-But-The-Top — https://arxiv.org/abs/1702.01417
6. Su et al. 2021, Whitening Sentence Representations — https://ar5iv.labs.arxiv.org/html/2103.15316
7. Ethayarajh 2019, Contextual Embedding Anisotropy — https://kawine.github.io/blog/nlp/2020/02/03/contextual.html
8. Stefanovitch et al. 2023, Graph+Community Detection — https://dl.acm.org/doi/fullHtml/10.1145/3543873.3587627
9. BERTopic documentation — https://maartengr.github.io/BERTopic/
10. sklearn clustering docs — https://scikit-learn.org/stable/modules/clustering.html
11. scipy linkage docs — https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
12. SBERT clustering examples — https://sbert.net/examples/sentence_transformer/applications/clustering/README.html
13. UMAP clustering docs — https://umap-learn.readthedocs.io/en/latest/clustering.html
14. Steck et al. 2024, Cosine Similarity Critique — https://arxiv.org/html/2403.05440v1
15. Moura et al. 2023, Transformer Intent Clustering — https://www.mdpi.com/2076-3417/13/8/5178
16. BERTopic FAQ (small datasets) — https://maartengr.github.io/BERTopic/faq.html
17. sentence-transformers community_detection — https://sbert.net/examples/sentence_transformer/applications/clustering/README.html
