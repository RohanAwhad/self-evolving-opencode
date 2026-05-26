# 006 — Goal Clustering (`goal_clusterer.py`)

## Purpose

Clusters goal strings by semantic similarity using sentence-transformer embeddings and agglomerative hierarchical clustering.

## API

### `cluster_goals(goals, *, model_name="all-mpnet-base-v2", min_cluster_size=3, max_cluster_size=20, distance_threshold=None) → ClusterResult`

## Pipeline

```
goals (strings)
  → _embed()          sentence-transformers → mean-subtract → L2-normalize
  → cosine_distances()  pairwise cosine distance matrix
  → linkage(average)    agglomerative clustering
  → _auto_cut()         silhouette score optimization (or distance_threshold)
  → _merge_small()       absorb clusters < min_cluster_size
  → _bisect_large()      recursively split clusters > max_cluster_size
  → _renumber()          contiguous labels from 0
  → ClusterResult
```

## Data Type

```python
@dataclass
class ClusterResult:
    clusters: dict[int, list[str]]  # cluster_id → goals
    labels: list[int]               # per-goal cluster label
```

## Internal Functions

### `_embed(goals, model_name) → np.ndarray`
Embed goals with sentence-transformers. Mean-subtract embeddings to remove shared domain signal. L2-normalize. Zero norms handled (set to 1 to avoid division by zero).

### `_auto_cut(Z, dist_matrix) → np.ndarray`
Finds the distance threshold that maximizes silhouette score. Tries 50 candidate thresholds from linkage merge distances. Enforces min_k=2, max_k=sqrt(n)+5. Falls back to 2 clusters if no score > 0.

### `_merge_small(embeddings, labels, min_size) → np.ndarray`
Iteratively merges clusters smaller than `min_size` into nearest neighbor cluster (by centroid cosine distance). If all clusters are small, collapses to single cluster.

### `_bisect_large(embeddings, labels, max_size, min_size) → np.ndarray`
Recursively bisects clusters larger than `max_size` using agglomerative sub-clustering on each oversized cluster. Splits into `ceil(size/max_size)` pieces.

### `_renumber(labels) → np.ndarray`
Maps cluster labels to contiguous 0-based integers.

## Edge Cases
- Empty goals → `ClusterResult({}, [])`
- Single goal → single cluster
- All identical goals → one cluster
- 2 goals → one cluster (below min_cluster_size, merged)

## Standalone CLI

```bash
uv run python -m src.goal_clusterer "goal 1" "goal 2" ...
uv run python -m src.goal_clusterer --min-cluster-size 5 --max-cluster-size 50
# Or via stdin:
echo '["goal 1", "goal 2"]' | uv run python -m src.goal_clusterer
```

## Dependencies
- sentence-transformers (embeddings)
- scipy (linkage, fcluster, squareform)
- sklearn (cosine_distances, silhouette_score)
- numpy (all array ops)

## Testing (`tests/test_goal_clusterer.py` — 24 tests)

**Pure unit tests** — no external dependencies, deterministic.

**Fixture**: List of ~30 goal strings with known semantic groupings (e.g., 10 about refactoring, 10 about testing, 10 about deployment).

**Coverage**:
- `_embed`: returns correct shape `(n, dim)`, deterministic (same input → same output), handles single string, handles empty list
- `_auto_cut`: finds clustering with positive silhouette score, falls back when no good cut found
- `_merge_small`: small cluster absorbed into nearest; all clusters small → collapse to 0; already-above-threshold untouched
- `_bisect_large`: oversize clusters split; result clusters all ≤ max_size; under-size clusters untouched; recursive splitting
- `_renumber`: `[3,7,3,12]` → `[0,1,0,2]`; already contiguous unchanged; single label
- `cluster_goals` end-to-end: known groupings cluster together; single goal → 1 cluster; empty → empty; all identical → 1 cluster; custom min/max sizes respected; `distance_threshold` mode works
