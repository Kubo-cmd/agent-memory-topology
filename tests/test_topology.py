#!/usr/bin/env python3
"""Tests for agent-memory-topology."""

from __future__ import annotations

import numpy as np
import pytest

from agent_memory_topology import (
    MemoryTopologyAnalyzer,
    TopologicalFeature,
    TopologyResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_data() -> np.ndarray:
    """Three well-separated clusters."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=5.0, scale=0.3, size=(20, 5))
    c2 = rng.normal(loc=-5.0, scale=0.3, size=(20, 5))
    c3 = rng.normal(loc=0.0, scale=0.3, size=(20, 5)) + np.array([0, 5, 0, 0, 0])
    return np.vstack([c1, c2, c3])


@pytest.fixture
def loop_data() -> np.ndarray:
    """Points on a circle (topological loop)."""
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    return np.column_stack([
        np.cos(theta) * 3,
        np.sin(theta) * 3,
        np.zeros(50),
        np.zeros(50),
        np.zeros(50),
    ])


@pytest.fixture
def mixed_data(cluster_data: np.ndarray, loop_data: np.ndarray) -> np.ndarray:
    """Clusters + loop combined."""
    return np.vstack([cluster_data, loop_data])


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestTopologicalFeature:
    """Test TopologicalFeature dataclass."""

    def test_valid_feature(self) -> None:
        feat = TopologicalFeature(
            dimension=0, birth=0.0, death=1.0,
            persistence=1.0, indices=[0, 1],
        )
        assert feat.dimension == 0
        assert feat.persistence == 1.0

    def test_negative_persistence_raises(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            TopologicalFeature(
                dimension=0, birth=1.0, death=0.5,
                persistence=-0.5, indices=[0],
            )

    def test_death_before_birth_raises(self) -> None:
        with pytest.raises(ValueError, match="Death"):
            TopologicalFeature(
                dimension=0, birth=2.0, death=1.0,
                persistence=1.0, indices=[0],
            )


class TestTopologyResult:
    """Test TopologyResult dataclass."""

    def test_summary_empty(self) -> None:
        result = TopologyResult(n_memories=10)
        summary = result.summary()
        assert "10 memories" in summary

    def test_summary_with_features(self) -> None:
        result = TopologyResult(
            n_memories=50,
            n_features={0: 3, 1: 1, 2: 0},
            significant_features=[
                TopologicalFeature(
                    dimension=0, birth=0.0, death=2.0,
                    persistence=2.0, indices=[0, 1],
                ),
            ],
        )
        summary = result.summary()
        assert "H0" in summary
        assert "H1" in summary


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Test input validation."""

    def test_empty_raises(self) -> None:
        analyzer = MemoryTopologyAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze(np.array([]).reshape(0, 5))

    def test_1d_raises(self) -> None:
        analyzer = MemoryTopologyAnalyzer()
        with pytest.raises(ValueError, match="2D"):
            analyzer.analyze(np.array([1, 2, 3]))

    def test_nan_raises(self) -> None:
        analyzer = MemoryTopologyAnalyzer()
        data = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        with pytest.raises(ValueError, match="non-finite"):
            analyzer.analyze(data)

    def test_inf_raises(self) -> None:
        analyzer = MemoryTopologyAnalyzer()
        data = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])
        with pytest.raises(ValueError, match="non-finite"):
            analyzer.analyze(data)

    def test_too_few_points_warns(self) -> None:
        analyzer = MemoryTopologyAnalyzer()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.warns(UserWarning, match="requires >= 3"):
            result = analyzer.analyze(data)
        assert result.n_memories == 2


# ---------------------------------------------------------------------------
# Core analysis tests
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Test topological analysis."""

    def test_detects_clusters(self, cluster_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer(max_dimension=0, persistence_threshold=0.5)
        result = analyzer.analyze(cluster_data)
        assert result.n_memories == 60
        assert 0 in result.n_features
        assert len(result.significant_features) > 0

    def test_detects_loop(self, loop_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer(max_dimension=1, persistence_threshold=0.5)
        result = analyzer.analyze(loop_data)
        assert result.n_memories == 50
        assert 1 in result.n_features
        assert result.n_features[1] > 0

    def test_mixed_analysis(self, mixed_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer(max_dimension=2, persistence_threshold=0.3)
        result = analyzer.analyze(mixed_data)
        assert result.n_memories == 110
        assert len(result.persistence_diagrams) > 0

    def test_labels_propagated(self, cluster_data: np.ndarray) -> None:
        labels = [f"memory_{i}" for i in range(len(cluster_data))]
        analyzer = MemoryTopologyAnalyzer(persistence_threshold=0.5)
        result = analyzer.analyze(cluster_data, labels=labels)
        # Features should have labels from the data
        labeled = [f for f in result.significant_features if f.label]
        assert len(labeled) > 0

    def test_higher_threshold_fewer_features(self, mixed_data: np.ndarray) -> None:
        low = MemoryTopologyAnalyzer(persistence_threshold=0.1).analyze(mixed_data)
        high = MemoryTopologyAnalyzer(persistence_threshold=1.0).analyze(mixed_data)
        assert len(high.significant_features) <= len(low.significant_features)


# ---------------------------------------------------------------------------
# Compression tests
# ---------------------------------------------------------------------------


class TestCompress:
    """Test memory compression."""

    def test_compress_reduces_size(self, mixed_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer()
        compressed, indices = analyzer.compress(mixed_data, keep_ratio=0.3)
        assert len(compressed) < len(mixed_data)
        assert len(indices) == len(compressed)

    def test_compress_indices_valid(self, mixed_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer()
        _, indices = analyzer.compress(mixed_data, keep_ratio=0.5)
        assert all(0 <= i < len(mixed_data) for i in indices)
        assert len(set(indices)) == len(indices)  # No duplicates

    def test_compress_keep_ratio_1(self, mixed_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer()
        compressed, indices = analyzer.compress(mixed_data, keep_ratio=1.0)
        assert len(compressed) == len(mixed_data)

    def test_compress_invalid_ratio(self, mixed_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer()
        with pytest.raises(ValueError, match="keep_ratio"):
            analyzer.compress(mixed_data, keep_ratio=0.0)
        with pytest.raises(ValueError, match="keep_ratio"):
            analyzer.compress(mixed_data, keep_ratio=1.5)

    def test_compress_strategies(self, mixed_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer()
        for strategy in ("topological", "diversity", "hybrid"):
            compressed, indices = analyzer.compress(
                mixed_data, keep_ratio=0.3, strategy=strategy,
            )
            assert len(compressed) > 0
            assert len(indices) > 0

    def test_compress_unknown_strategy(self, mixed_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer()
        with pytest.raises(ValueError, match="Unknown strategy"):
            analyzer.compress(mixed_data, strategy="nonsense")


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------


class TestCompare:
    """Test topology comparison."""

    def test_identical_topologies(self, cluster_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer()
        distance = analyzer.compare(cluster_data, cluster_data.copy())
        assert distance == pytest.approx(0.0, abs=0.01)

    def test_different_topologies(
        self, cluster_data: np.ndarray, loop_data: np.ndarray,
    ) -> None:
        analyzer = MemoryTopologyAnalyzer()
        distance = analyzer.compare(cluster_data, loop_data)
        assert distance > 0.0


# ---------------------------------------------------------------------------
# Feature detection tests
# ---------------------------------------------------------------------------


class TestFeatureDetection:
    """Test specialized feature detectors."""

    def test_detect_reasoning_loops(self, loop_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer(max_dimension=1, persistence_threshold=0.3)
        loops = analyzer.detect_reasoning_loops(loop_data, min_persistence=0.3)
        assert isinstance(loops, list)
        for loop in loops:
            assert loop.dimension == 1
            assert loop.persistence >= 0.3

    def test_detect_knowledge_gaps(self, mixed_data: np.ndarray) -> None:
        analyzer = MemoryTopologyAnalyzer(max_dimension=2, persistence_threshold=0.1)
        gaps = analyzer.detect_knowledge_gaps(mixed_data, min_persistence=0.1)
        assert isinstance(gaps, list)
        for gap in gaps:
            assert gap.dimension == 2


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestConstructor:
    """Test analyzer initialization."""

    def test_default_params(self) -> None:
        analyzer = MemoryTopologyAnalyzer()
        assert analyzer.max_dimension == 2
        assert analyzer.persistence_threshold == 0.1
        assert analyzer.metric == "euclidean"

    def test_custom_params(self) -> None:
        analyzer = MemoryTopologyAnalyzer(
            max_dimension=1, persistence_threshold=0.5, metric="cosine",
        )
        assert analyzer.max_dimension == 1
        assert analyzer.persistence_threshold == 0.5

    def test_negative_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="max_dimension"):
            MemoryTopologyAnalyzer(max_dimension=-1)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="persistence_threshold"):
            MemoryTopologyAnalyzer(persistence_threshold=-0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
