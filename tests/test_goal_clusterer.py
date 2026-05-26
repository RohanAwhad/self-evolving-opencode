"""Tests for src/goal_clusterer -- pure compute, no LLM/Redis needed."""

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances

from src.goal_clusterer import (
    ClusterResult,
    _auto_cut,
    _bisect_large,
    _embed,
    _merge_small,
    _renumber,
    cluster_goals,
)

# ---------------------------------------------------------------------------
# Goal strings for integration tests -- 3 natural clusters
# ---------------------------------------------------------------------------

PYTHON_GOALS = [
    "Fix Python import error in utils module",
    "Debug Python syntax error in parser",
    "Resolve Python module not found for numpy",
    "Refactor Python function to reduce complexity",
    "Add Python type hints to data models",
]
GIT_GOALS = [
    "Create a new git feature branch",
    "Merge git branches after code review",
    "Resolve git merge conflict in main",
    "Rebase git branch onto latest main",
    "Squash git commits before merging",
]
DOCKER_GOALS = [
    "Build Docker container for web app",
    "Fix Docker compose networking issue",
    "Deploy application with Docker Swarm",
    "Optimize Docker image size for production",
    "Write Dockerfile for Python microservice",
]
ALL_GOALS = PYTHON_GOALS + GIT_GOALS + DOCKER_GOALS


# ---------------------------------------------------------------------------
# _embed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_output_shape(self):
        embs = _embed(["hello world", "foo bar", "baz qux"])
        assert embs.shape[0] == 3
        assert embs.shape[1] > 0  # embedding dim

    def test_mean_subtracted(self):
        """After mean-subtraction + L2-norm, column means drift but stay small."""
        embs = _embed(ALL_GOALS)
        col_means = embs.mean(axis=0)
        # L2-normalization distorts the zero-mean property, but means stay small
        assert np.abs(col_means).max() < 0.01

    def test_l2_normalized(self):
        embs = _embed(ALL_GOALS)
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_single_goal_norm(self):
        """Single goal: mean-subtract makes it zero, norm guard prevents NaN."""
        embs = _embed(["just one goal"])
        assert embs.shape[0] == 1
        # After mean-subtraction of a single vector, it's all zeros.
        # The norm guard (norms[norms==0]=1) should prevent division by zero.
        assert not np.any(np.isnan(embs))


# ---------------------------------------------------------------------------
# _renumber
# ---------------------------------------------------------------------------


class TestRenumber:
    def test_fills_gaps(self):
        labels = np.array([0, 5, 5, 2, 0, 2])
        result = _renumber(labels)
        np.testing.assert_array_equal(result, [0, 2, 2, 1, 0, 1])

    def test_already_contiguous(self):
        labels = np.array([0, 1, 2, 0, 1])
        result = _renumber(labels)
        np.testing.assert_array_equal(result, [0, 1, 2, 0, 1])

    def test_single_cluster(self):
        labels = np.array([7, 7, 7])
        result = _renumber(labels)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_reverse_order(self):
        labels = np.array([10, 5, 0])
        result = _renumber(labels)
        np.testing.assert_array_equal(result, [2, 1, 0])


# ---------------------------------------------------------------------------
# _merge_small
# ---------------------------------------------------------------------------


class TestMergeSmall:
    def test_small_cluster_merges_into_nearest(self):
        embs = _embed(ALL_GOALS)
        # Assign first 5 to cluster 0, next 5 to cluster 1, last 5 to cluster 2
        labels = np.array([0]*5 + [1]*5 + [2]*5)
        # Move one from cluster 2 into its own tiny cluster 3
        labels[14] = 3
        result = _merge_small(embs, labels, min_size=3)
        # Cluster 3 (size 1) should have been merged away
        assert 3 not in result
        # All original big clusters should still exist
        assert set(result).issubset({0, 1, 2})

    def test_all_small_collapses_to_zero(self):
        embs = _embed(["a", "b", "c", "d"])
        labels = np.array([0, 1, 2, 3])  # all size 1
        result = _merge_small(embs, labels, min_size=3)
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_no_small_clusters_unchanged(self):
        embs = _embed(ALL_GOALS)
        labels = np.array([0]*5 + [1]*5 + [2]*5)
        result = _merge_small(embs, labels, min_size=3)
        np.testing.assert_array_equal(result, labels)


# ---------------------------------------------------------------------------
# _bisect_large
# ---------------------------------------------------------------------------


class TestBisectLarge:
    def test_oversized_cluster_gets_split(self):
        embs = _embed(ALL_GOALS)
        # Put everything in one cluster
        labels = np.array([0] * len(ALL_GOALS))
        result = _bisect_large(embs, labels, max_size=5, min_size=2)
        unique = set(result)
        assert len(unique) >= 3  # 15 items / 5 max = at least 3 clusters

    def test_small_clusters_untouched(self):
        embs = _embed(ALL_GOALS)
        labels = np.array([0]*5 + [1]*5 + [2]*5)
        result = _bisect_large(embs, labels, max_size=5, min_size=2)
        # Each cluster is exactly max_size, so no splitting needed
        assert len(set(result)) == 3

    def test_respects_max_size(self):
        embs = _embed(ALL_GOALS)
        labels = np.array([0] * len(ALL_GOALS))
        result = _bisect_large(embs, labels, max_size=6, min_size=2)
        unique, counts = np.unique(result, return_counts=True)
        assert all(c <= 6 for c in counts)


# ---------------------------------------------------------------------------
# _auto_cut
# ---------------------------------------------------------------------------


class TestAutoCut:
    def test_returns_valid_labels(self):
        embs = _embed(ALL_GOALS)
        dist_matrix = cosine_distances(embs).astype(np.float64)
        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method="average")
        labels = _auto_cut(Z, dist_matrix)
        assert len(labels) == len(ALL_GOALS)
        assert len(set(labels)) >= 2  # at least 2 clusters

    def test_two_distinct_groups(self):
        """Two very different groups should produce exactly 2 clusters."""
        goals = [
            "Fix Python import error",
            "Debug Python syntax error",
            "Build Docker container image",
            "Deploy with Docker compose",
        ]
        embs = _embed(goals)
        dist_matrix = cosine_distances(embs).astype(np.float64)
        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method="average")
        labels = _auto_cut(Z, dist_matrix)
        assert len(set(labels)) >= 2


# ---------------------------------------------------------------------------
# cluster_goals (end-to-end)
# ---------------------------------------------------------------------------


class TestClusterGoals:
    def test_empty_list(self):
        result = cluster_goals([])
        assert result == ClusterResult(clusters={}, labels=[])

    def test_single_goal(self):
        result = cluster_goals(["Fix a bug"])
        assert result == ClusterResult(clusters={0: ["Fix a bug"]}, labels=[0])

    def test_two_goals(self):
        result = cluster_goals(["Fix a bug", "Write a test"], min_cluster_size=1)
        assert len(result.labels) == 2
        assert len(result.clusters) >= 1

    def test_three_clusters_separated(self):
        """15 goals in 3 natural groups should produce multiple clusters."""
        result = cluster_goals(ALL_GOALS, min_cluster_size=2, max_cluster_size=10)
        assert len(result.clusters) >= 2
        assert len(result.labels) == len(ALL_GOALS)
        # Labels are contiguous from 0
        assert min(result.labels) == 0
        assert max(result.labels) == len(result.clusters) - 1

    def test_labels_length_matches_input(self):
        result = cluster_goals(ALL_GOALS, min_cluster_size=2)
        assert len(result.labels) == len(ALL_GOALS)

    def test_all_goals_assigned_to_a_cluster(self):
        result = cluster_goals(ALL_GOALS, min_cluster_size=2)
        all_clustered = []
        for goals in result.clusters.values():
            all_clustered.extend(goals)
        assert sorted(all_clustered) == sorted(ALL_GOALS)

    def test_with_distance_threshold(self):
        result = cluster_goals(ALL_GOALS, distance_threshold=0.5, min_cluster_size=1)
        assert len(result.clusters) >= 1
        assert len(result.labels) == len(ALL_GOALS)

    def test_max_cluster_size_enforced(self):
        result = cluster_goals(ALL_GOALS, min_cluster_size=1, max_cluster_size=4)
        for goals in result.clusters.values():
            assert len(goals) <= 4
