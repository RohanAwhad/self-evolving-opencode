"""Cluster goal strings using embeddings + agglomerative clustering.

Pipeline: embed → mean-subtract → L2-normalize → cosine distance → average-linkage
→ fcluster → bisect oversized clusters.
"""

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances


@dataclass
class ClusterResult:
    """Output of goal clustering."""

    clusters: dict[int, list[str]]  # cluster_id -> goals
    labels: list[int]  # per-goal cluster label (aligned with input)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "all-mpnet-base-v2"
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _embed(goals: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed, mean-subtract (removes shared domain signal), then L2-normalize."""
    model = _get_model(model_name)
    embeddings = model.encode(goals, show_progress_bar=False, convert_to_numpy=True)
    # Mean subtraction — critical for narrow-domain text
    centered = embeddings - embeddings.mean(axis=0)
    # L2-normalize
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return centered / norms


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------


def cluster_goals(
    goals: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    min_cluster_size: int = 3,
    max_cluster_size: int = 20,
    distance_threshold: float | None = None,
) -> ClusterResult:
    """Cluster goal strings into similarity buckets.

    Uses agglomerative clustering (average linkage, cosine distance) with
    mean-subtracted embeddings. Automatically selects the number of clusters
    via silhouette score optimization unless distance_threshold is provided.

    Args:
        goals: Raw goal strings.
        model_name: Sentence-transformer model for embeddings.
        min_cluster_size: Clusters smaller than this get merged into nearest
            neighbor cluster.
        max_cluster_size: Clusters larger than this get bisected recursively.
        distance_threshold: If set, cut the dendrogram at this cosine distance
            instead of optimizing silhouette score. Range [0, 2].

    Returns:
        ClusterResult with cluster mapping and per-goal labels.
    """
    if not goals:
        return ClusterResult(clusters={}, labels=[])
    if len(goals) == 1:
        return ClusterResult(clusters={0: goals}, labels=[0])

    embeddings = _embed(goals, model_name)
    dist_matrix = cosine_distances(embeddings).astype(np.float64)
    condensed = squareform(dist_matrix, checks=False)

    # --- agglomerative clustering (average linkage) ---
    Z = linkage(condensed, method="average")

    # --- choose cut point ---
    if distance_threshold is not None:
        labels = fcluster(Z, t=distance_threshold, criterion="distance")
    else:
        labels = _auto_cut(Z, dist_matrix)

    # fcluster labels start at 1; shift to 0-based
    labels = labels - 1

    # --- merge small clusters into nearest neighbor ---
    labels = _merge_small(embeddings, labels, min_cluster_size)

    # --- bisect oversized clusters ---
    labels = _bisect_large(embeddings, labels, max_cluster_size, min_cluster_size)

    # --- renumber contiguously ---
    labels = _renumber(labels)

    # --- build output ---
    clusters: dict[int, list[str]] = {}
    for goal, cid in zip(goals, labels):
        clusters.setdefault(int(cid), []).append(goal)

    return ClusterResult(clusters=clusters, labels=[int(l) for l in labels])


# ---------------------------------------------------------------------------
# Auto-cut: silhouette score optimization
# ---------------------------------------------------------------------------


def _auto_cut(Z: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
    """Find the distance threshold that maximizes silhouette score."""
    n = dist_matrix.shape[0]
    max_k = min(n - 1, int(np.sqrt(n)) + 5)
    min_k = 2

    best_score = -1.0
    best_labels = np.zeros(n, dtype=int)

    # Try a range of thresholds by sampling the merge distances in the linkage
    merge_dists = Z[:, 2]
    candidates = np.linspace(merge_dists.min(), merge_dists.max(), num=50)

    for t in candidates:
        labels = fcluster(Z, t=t, criterion="distance")
        k = len(set(labels))
        if k < min_k or k > max_k:
            continue
        score = silhouette_score(dist_matrix, labels, metric="precomputed")
        if score > best_score:
            best_score = score
            best_labels = labels.copy()

    # Fallback: if nothing scored well, use 2 clusters
    if best_score <= 0:
        best_labels = fcluster(Z, t=2, criterion="maxclust")

    return best_labels


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _merge_small(
    embeddings: np.ndarray, labels: np.ndarray, min_size: int
) -> np.ndarray:
    """Merge clusters smaller than min_size into their nearest neighbor."""
    labels = labels.copy()
    while True:
        unique, counts = np.unique(labels, return_counts=True)
        small = [int(u) for u, c in zip(unique, counts) if c < min_size]
        big = {int(u) for u, c in zip(unique, counts) if c >= min_size}
        if not small:
            break
        if not big:
            labels[:] = 0
            break
        for cid in small:
            mask = labels == cid
            centroid = embeddings[mask].mean(axis=0)
            best_id = -1
            best_dist = float("inf")
            for bid in big:
                bmask = labels == bid
                bcentroid = embeddings[bmask].mean(axis=0)
                dist = cosine_distances(
                    centroid.reshape(1, -1), bcentroid.reshape(1, -1)
                )[0, 0]
                if dist < best_dist:
                    best_dist = dist
                    best_id = bid
            labels[mask] = best_id
    return labels


def _bisect_large(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_size: int,
    min_size: int,
) -> np.ndarray:
    """Recursively bisect clusters larger than max_size using agglomerative sub-clustering."""
    labels = labels.copy()
    next_id = int(labels.max()) + 1

    changed = True
    while changed:
        changed = False
        for cid in list(np.unique(labels)):
            mask = labels == cid
            size = int(mask.sum())
            if size <= max_size:
                continue

            # Sub-cluster via agglomerative bisection
            sub_emb = embeddings[mask]
            sub_dist = cosine_distances(sub_emb).astype(np.float64)
            sub_condensed = squareform(sub_dist, checks=False)
            sub_Z = linkage(sub_condensed, method="average")

            # Try to split into k pieces where each piece <= max_size
            target_k = max(2, (size + max_size - 1) // max_size)
            sub_labels = fcluster(sub_Z, t=target_k, criterion="maxclust")

            # Check we actually split (not all same label)
            if len(set(sub_labels)) < 2:
                continue

            # Remap to global IDs
            idxs = np.where(mask)[0]
            remap: dict[int, int] = {}
            for sl in np.unique(sub_labels):
                remap[int(sl)] = next_id
                next_id += 1
            for i, sl in zip(idxs, sub_labels):
                labels[i] = remap[int(sl)]

            changed = True

    return labels


def _renumber(labels: np.ndarray) -> np.ndarray:
    """Renumber labels contiguously starting from 0."""
    mapping = {old: new for new, old in enumerate(sorted(set(labels)))}
    return np.array([mapping[int(l)] for l in labels])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster goal strings into similarity buckets."
    )
    parser.add_argument(
        "goals",
        nargs="*",
        help="Goal strings. If omitted, reads JSON array from stdin.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence-transformer model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="Minimum cluster size; smaller clusters merge (default: 3)",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=20,
        help="Maximum cluster size; larger clusters bisect (default: 20)",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=None,
        help="Cut dendrogram at this cosine distance (default: auto via silhouette)",
    )
    args = parser.parse_args()

    if args.goals:
        goals = args.goals
    else:
        raw = sys.stdin.read().strip()
        goals = json.loads(raw)

    result = cluster_goals(
        goals,
        model_name=args.model,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        distance_threshold=args.distance_threshold,
    )

    output = {
        "num_clusters": len(result.clusters),
        "clusters": {
            str(k): {"size": len(v), "goals": v}
            for k, v in sorted(result.clusters.items())
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
