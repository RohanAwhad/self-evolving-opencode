"""Cluster a list of goal strings into similarity buckets using embeddings + HDBSCAN."""

import argparse
import json
import sys
from dataclasses import dataclass, field

import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances


@dataclass
class ClusterResult:
    """Output of goal clustering."""

    clusters: dict[int, list[str]]  # cluster_id -> goals
    labels: list[int]  # per-goal cluster label (aligned with input)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _embed(goals: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(goals, show_progress_bar=False, convert_to_numpy=True)


def _nearest_cluster(
    point: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    valid_ids: set[int],
) -> int:
    """Return the cluster id from *valid_ids* whose centroid is closest to *point*."""
    best_id = -1
    best_dist = float("inf")
    for cid in valid_ids:
        mask = labels == cid
        centroid = embeddings[mask].mean(axis=0)
        dist = cosine_distances(point.reshape(1, -1), centroid.reshape(1, -1))[0, 0]
        if dist < best_dist:
            best_dist = dist
            best_id = cid
    return best_id


def cluster_goals(
    goals: list[str],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    min_cluster_size: int = 5,
    max_cluster_size: int = 100,
    hdbscan_min_samples: int | None = None,
) -> ClusterResult:
    """Cluster goal strings into similarity buckets.

    Args:
        goals: Raw goal strings.
        model_name: Sentence-transformer model for embeddings.
        min_cluster_size: Clusters smaller than this get merged into their
            nearest neighbour cluster.
        max_cluster_size: Clusters larger than this get split via recursive
            sub-clustering.
        hdbscan_min_samples: HDBSCAN min_samples param. Defaults to
            min_cluster_size if not set.

    Returns:
        ClusterResult with cluster mapping and per-goal labels.
    """
    if not goals:
        return ClusterResult(clusters={}, labels=[])

    embeddings = _embed(goals, model_name)

    # --- initial clustering ---
    min_samples = hdbscan_min_samples or min_cluster_size
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)

    # --- absorb noise (-1) into nearest real cluster ---
    real_ids = {int(l) for l in labels if l != -1}
    if not real_ids:
        # everything is noise – put it all in one bucket
        labels[:] = 0
        real_ids = {0}
    else:
        noise_mask = labels == -1
        for idx in np.where(noise_mask)[0]:
            labels[idx] = _nearest_cluster(
                embeddings[idx], embeddings, labels, real_ids
            )

    # --- merge small clusters ---
    labels = _merge_small(embeddings, labels, min_cluster_size)

    # --- split large clusters ---
    labels = _split_large(embeddings, labels, max_cluster_size, min_cluster_size)

    # --- renumber contiguously from 0 ---
    labels = _renumber(labels)

    # --- build output ---
    clusters: dict[int, list[str]] = {}
    for goal, cid in zip(goals, labels):
        clusters.setdefault(int(cid), []).append(goal)

    return ClusterResult(clusters=clusters, labels=[int(l) for l in labels])


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _merge_small(
    embeddings: np.ndarray, labels: np.ndarray, min_size: int
) -> np.ndarray:
    """Merge clusters with fewer than *min_size* members into nearest bigger cluster."""
    labels = labels.copy()
    while True:
        unique, counts = np.unique(labels, return_counts=True)
        small = [int(u) for u, c in zip(unique, counts) if c < min_size]
        big = {int(u) for u, c in zip(unique, counts) if c >= min_size}
        if not small:
            break
        if not big:
            # everything is small – collapse into one cluster
            labels[:] = 0
            break
        for cid in small:
            mask = labels == cid
            centroid = embeddings[mask].mean(axis=0)
            target = _nearest_cluster(centroid, embeddings, labels, big)
            labels[mask] = target
    return labels


def _split_large(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_size: int,
    min_size: int,
) -> np.ndarray:
    """Split clusters larger than *max_size* via recursive HDBSCAN."""
    labels = labels.copy()
    next_id = int(labels.max()) + 1

    for cid in list(np.unique(labels)):
        mask = labels == cid
        if mask.sum() <= max_size:
            continue

        sub_emb = embeddings[mask]
        sub_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            metric="euclidean",
        )
        sub_labels = sub_clusterer.fit_predict(sub_emb)

        # absorb sub-noise into nearest sub-cluster
        sub_real = {int(l) for l in sub_labels if l != -1}
        if not sub_real:
            continue  # can't split further
        for i in np.where(sub_labels == -1)[0]:
            sub_labels[i] = _nearest_cluster(
                sub_emb[i], sub_emb, sub_labels, sub_real
            )

        # remap sub-labels to global ids
        remap = {}
        for sl in np.unique(sub_labels):
            remap[int(sl)] = next_id
            next_id += 1
        idxs = np.where(mask)[0]
        for i, sl in zip(idxs, sub_labels):
            labels[i] = remap[int(sl)]

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
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size; smaller clusters merge (default: 5)",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=100,
        help="Maximum cluster size; larger clusters split (default: 100)",
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
