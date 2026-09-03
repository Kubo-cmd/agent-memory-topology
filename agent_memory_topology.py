#!/usr/bin/env python3
"""
Agent Memory Topology — Semantic compression via persistent homology.

Uses topological data analysis (TDA) to detect structural features in agent
memory embeddings: clusters (H0), reasoning loops (H1), knowledge gaps (H2).
Compresses memory by preserving topology, not just similarity.

Uses algebraic topology to model structural patterns in agent memory.

Theory:
- Memory embeddings form a point cloud in high-dimensional space
- Persistent homology detects topological features at multiple scales
- H0: connected components (conceptual clusters)
- H1: loops (reasoning cycles, recurring patterns)
- H2: voids (knowledge gaps, missing concepts)
- Persistence diagram shows which features are "real" vs noise
- Compress by keeping only topologically significant memories

Usage:
    from agent_memory_topology import MemoryTopologyAnalyzer

    analyzer = MemoryTopologyAnalyzer()
    result = analyzer.analyze(memory_embeddings)
    compressed = analyzer.compress(memory_embeddings, keep_ratio=0.3)
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any

import numpy as np
from ripser import ripser

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


def _require_aware(value: datetime, field_name: str) -> None:
    """Reject ambiguous naive timestamps at the public API boundary."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


def _freeze_value(value: Any) -> Any:
    """Recursively detach and freeze common attribute containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_attributes(attributes: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable copy of an attribute mapping."""
    return MappingProxyType({key: _freeze_value(value) for key, value in attributes.items()})


@dataclass(frozen=True)
class ProvenanceRecord:
    """Evidence describing where and when a relationship was observed."""

    source_id: str
    source_type: str
    observed_at: datetime
    evidence_id: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id cannot be empty")
        if not self.source_type.strip():
            raise ValueError("source_type cannot be empty")
        _require_aware(self.observed_at, "observed_at")
        object.__setattr__(self, "attributes", _freeze_attributes(self.attributes))


@dataclass(frozen=True)
class MemoryRelationship:
    """A provenance-backed, bitemporal edge in a semantic memory graph.

    ``valid_from``/``valid_until`` describe when the claim is true in the
    represented world. ``recorded_at``/``retracted_at`` describe when the
    memory system knew the claim. Both intervals are half-open.
    """

    subject_id: str
    predicate: str
    object_id: str
    valid_from: datetime
    recorded_at: datetime
    valid_until: datetime | None = None
    retracted_at: datetime | None = None
    confidence: float = 1.0
    provenance: tuple[ProvenanceRecord, ...] = field(default_factory=tuple)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.subject_id, "subject_id"),
            (self.predicate, "predicate"),
            (self.object_id, "object_id"),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.provenance:
            raise ValueError("provenance must contain at least one record")
        if not all(isinstance(item, ProvenanceRecord) for item in self.provenance):
            raise TypeError("provenance entries must be ProvenanceRecord instances")

        _require_aware(self.valid_from, "valid_from")
        _require_aware(self.recorded_at, "recorded_at")
        if self.valid_until is not None:
            _require_aware(self.valid_until, "valid_until")
            if self.valid_until <= self.valid_from:
                raise ValueError("valid_until must be later than valid_from")
        if self.retracted_at is not None:
            _require_aware(self.retracted_at, "retracted_at")
            if self.retracted_at <= self.recorded_at:
                raise ValueError("retracted_at must be later than recorded_at")

        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "attributes", _freeze_attributes(self.attributes))

    def is_valid_at(
        self,
        valid_at: datetime,
        known_at: datetime | None = None,
    ) -> bool:
        """Return whether this edge is valid, optionally as known at a time."""
        _require_aware(valid_at, "valid_at")
        if known_at is not None:
            _require_aware(known_at, "known_at")
        if not (
            self.valid_from <= valid_at
            and (self.valid_until is None or valid_at < self.valid_until)
        ):
            return False
        if known_at is None:
            return self.retracted_at is None

        return self.recorded_at <= known_at and (
            self.retracted_at is None or known_at < self.retracted_at
        )


class TemporalRelationshipGraph:
    """Query semantic memory edges without discarding historical truth."""

    def __init__(self, relationships: Iterable[MemoryRelationship] = ()) -> None:
        self._relationships: list[MemoryRelationship] = []
        for relationship in relationships:
            self.add(relationship)

    def add(self, relationship: MemoryRelationship) -> None:
        """Add a validated relationship to the graph."""
        if not isinstance(relationship, MemoryRelationship):
            raise TypeError("relationship must be a MemoryRelationship")
        self._relationships.append(relationship)

    def retract(
        self,
        relationship: MemoryRelationship,
        retracted_at: datetime,
    ) -> MemoryRelationship:
        """Retract an edge while retaining its earlier knowledge-time history."""
        if not isinstance(relationship, MemoryRelationship):
            raise TypeError("relationship must be a MemoryRelationship")
        _require_aware(retracted_at, "retracted_at")
        for index, candidate in enumerate(self._relationships):
            if candidate is relationship:
                if candidate.retracted_at is not None:
                    raise ValueError("relationship is already retracted")
                retracted = replace(candidate, retracted_at=retracted_at)
                self._relationships[index] = retracted
                return retracted
        raise ValueError("relationship is not present in this graph")

    def relationships_at(
        self,
        valid_at: datetime,
        known_at: datetime | None = None,
        predicate: str | None = None,
    ) -> list[MemoryRelationship]:
        """Return edges valid at a world time and optional knowledge time."""
        _require_aware(valid_at, "valid_at")
        if known_at is not None:
            _require_aware(known_at, "known_at")
        return [
            relationship
            for relationship in self._relationships
            if (predicate is None or relationship.predicate == predicate)
            and relationship.is_valid_at(valid_at, known_at)
        ]

    def neighbors(
        self,
        memory_id: str,
        valid_at: datetime,
        known_at: datetime | None = None,
        direction: str = "both",
    ) -> list[str]:
        """Return deterministic neighboring IDs for a temporal graph slice."""
        if direction not in {"incoming", "outgoing", "both"}:
            raise ValueError("direction must be incoming, outgoing, or both")

        neighbors = set()
        for relationship in self.relationships_at(valid_at, known_at):
            if direction in {"outgoing", "both"} and relationship.subject_id == memory_id:
                neighbors.add(relationship.object_id)
            if direction in {"incoming", "both"} and relationship.object_id == memory_id:
                neighbors.add(relationship.subject_id)
        return sorted(neighbors)


@dataclass
class TopologicalFeature:
    """A detected topological feature in memory space."""

    dimension: int  # 0=cluster, 1=loop, 2=void
    birth: float  # Scale at which feature appears
    death: float  # Scale at which feature disappears
    persistence: float  # death - birth (significance)
    indices: list[int]  # Memory indices involved
    label: str = ""  # Optional semantic label

    def __post_init__(self) -> None:
        if self.persistence < 0:
            raise ValueError(f"Persistence cannot be negative: {self.persistence}")
        if self.death < self.birth:
            raise ValueError(f"Death ({self.death}) < birth ({self.birth})")


@dataclass
class TopologyResult:
    """Result of topological analysis on memory embeddings."""

    n_memories: int
    n_features: dict[int, int] = field(default_factory=dict)  # dim -> count
    significant_features: list[TopologicalFeature] = field(default_factory=list)
    persistence_diagrams: dict[int, np.ndarray] = field(default_factory=dict)
    bottleneck_distance: float | None = None  # Comparison metric
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of topological features."""
        lines = [f"Memory Topology Analysis ({self.n_memories} memories)"]
        lines.append("=" * 60)

        if 0 in self.n_features:
            lines.append(f"  H0 (clusters): {self.n_features[0]} connected components")
        if 1 in self.n_features:
            lines.append(f"  H1 (loops): {self.n_features[1]} reasoning cycles detected")
        if 2 in self.n_features:
            lines.append(f"  H2 (voids): {self.n_features[2]} knowledge gaps")

        if self.significant_features:
            lines.append("\nTop 5 most persistent features:")
            sorted_features = sorted(
                self.significant_features,
                key=lambda f: f.persistence,
                reverse=True,
            )[:5]
            for i, feat in enumerate(sorted_features, 1):
                dim_name = {0: "cluster", 1: "loop", 2: "void"}.get(
                    feat.dimension, f"H{feat.dimension}"
                )
                lines.append(
                    f"  {i}. {dim_name} (persistence={feat.persistence:.3f}, "
                    f"birth={feat.birth:.3f}, death={feat.death:.3f})"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------


class MemoryTopologyAnalyzer:
    """Analyze and compress agent memory using persistent homology.

    This applies algebraic topology to memory embeddings to detect structural
    features that similarity-based methods miss: reasoning loops, conceptual
    clusters, and knowledge gaps.

    Args:
        max_dimension: Maximum homology dimension to compute (default 2).
                       0=clusters, 1=loops, 2=voids.
        persistence_threshold: Minimum persistence to consider a feature
                               significant (default 0.1).
        metric: Distance metric for point cloud (default 'euclidean').

    Example:
        >>> analyzer = MemoryTopologyAnalyzer()
        >>> result = analyzer.analyze(memory_embeddings)
        >>> print(result.summary())
        >>> compressed = analyzer.compress(memory_embeddings, keep_ratio=0.3)
    """

    def __init__(
        self,
        max_dimension: int = 2,
        persistence_threshold: float = 0.1,
        metric: str = "euclidean",
    ) -> None:
        if max_dimension < 0:
            raise ValueError(f"max_dimension must be >= 0, got {max_dimension}")
        if persistence_threshold < 0:
            raise ValueError(f"persistence_threshold must be >= 0, got {persistence_threshold}")

        self.max_dimension = max_dimension
        self.persistence_threshold = persistence_threshold
        self.metric = metric

    def analyze(
        self,
        embeddings: np.ndarray,
        labels: list[str] | None = None,
    ) -> TopologyResult:
        """Analyze topological structure of memory embeddings.

        Args:
            embeddings: Array of shape (n_memories, embedding_dim).
            labels: Optional semantic labels for each memory.

        Returns:
            TopologyResult with detected features and persistence diagrams.

        Raises:
            ValueError: If embeddings are invalid.
        """
        embeddings = self._validate_embeddings(embeddings)
        n_memories = len(embeddings)

        if n_memories < 3:
            warnings.warn(
                f"Only {n_memories} memories — topological analysis requires >= 3 points. "
                f"Returning empty result.",
                UserWarning,
                stacklevel=2,
            )
            return TopologyResult(
                n_memories=n_memories,
                n_features={},
                significant_features=[],
                persistence_diagrams={},
            )

        # Compute persistent homology
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ripser(
                embeddings,
                maxdim=self.max_dimension,
                metric=self.metric,
                do_cocycles=False,
            )

        diagrams = result["dgms"]

        # Extract features
        all_features: list[TopologicalFeature] = []
        n_features: dict[int, int] = {}

        # Build pairwise distance matrix for index mapping
        from scipy.spatial.distance import pdist, squareform

        dist_matrix = squareform(pdist(embeddings, metric=self.metric))

        for dim in range(self.max_dimension + 1):
            if dim >= len(diagrams):
                continue

            diagram = diagrams[dim]
            if len(diagram) == 0:
                n_features[dim] = 0
                continue

            # Filter out infinite persistence (birth=death=inf)
            finite_mask = np.isfinite(diagram).all(axis=1)
            finite_diagram = diagram[finite_mask]

            n_features[dim] = len(finite_diagram)

            # For each feature, find the memories most involved
            # H0: find the point closest to the cluster center
            # H1: find points on the loop (highest local density on cycle)
            # H2: find points bordering the void
            for _i, (birth, death) in enumerate(finite_diagram):
                persistence = death - birth
                if persistence >= self.persistence_threshold:
                    # Find representative indices for this feature
                    representative_indices = self._find_representative_indices(
                        dim,
                        birth,
                        death,
                        dist_matrix,
                        embeddings,
                    )
                    feature = TopologicalFeature(
                        dimension=dim,
                        birth=float(birth),
                        death=float(death),
                        persistence=float(persistence),
                        indices=representative_indices,
                        label=(
                            labels[representative_indices[0]]
                            if labels and representative_indices[0] < len(labels)
                            else ""
                        ),
                    )
                    all_features.append(feature)

        return TopologyResult(
            n_memories=n_memories,
            n_features=n_features,
            significant_features=all_features,
            persistence_diagrams=diagrams,
        )

    def compress(
        self,
        embeddings: np.ndarray,
        keep_ratio: float = 0.3,
        strategy: str = "topological",
    ) -> tuple[np.ndarray, list[int]]:
        """Compress memory by preserving topological structure.

        Args:
            embeddings: Array of shape (n_memories, embedding_dim).
            keep_ratio: Fraction of memories to keep (0.0 to 1.0).
            strategy: Compression strategy:
                      - "topological": Keep memories with high persistence
                      - "diversity": Keep one from each topological cluster
                      - "hybrid": Combine both (default)

        Returns:
            Tuple of (compressed_embeddings, kept_indices).

        Raises:
            ValueError: If parameters are invalid.
        """
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

        embeddings = self._validate_embeddings(embeddings)
        n_memories = len(embeddings)
        n_keep = max(1, int(n_memories * keep_ratio))

        if n_keep >= n_memories:
            return embeddings.copy(), list(range(n_memories))

        # Analyze topology
        result = self.analyze(embeddings)

        if strategy == "topological":
            kept_indices = self._compress_topological(result, n_keep)
        elif strategy == "diversity":
            kept_indices = self._compress_diversity(result, n_keep)
        elif strategy == "hybrid":
            topo_indices = self._compress_topological(result, n_keep // 2)
            diversity_indices = self._compress_diversity(result, n_keep - len(topo_indices))
            kept_indices = sorted(set(topo_indices) | set(diversity_indices))[:n_keep]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return embeddings[kept_indices], kept_indices

    def compare(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> float:
        """Compare topological structure of two memory sets.

        Computes bottleneck distance between persistence diagrams.
        Lower distance = more similar topology.

        Args:
            embeddings1: First memory set.
            embeddings2: Second memory set.

        Returns:
            Bottleneck distance (0.0 = identical topology).
        """
        embeddings1 = self._validate_embeddings(embeddings1)
        embeddings2 = self._validate_embeddings(embeddings2)

        result1 = self.analyze(embeddings1)
        result2 = self.analyze(embeddings2)

        # Compute bottleneck distance for each dimension
        distances = []
        for dim in range(self.max_dimension + 1):
            if dim < len(result1.persistence_diagrams) and dim < len(result2.persistence_diagrams):
                diag1 = result1.persistence_diagrams[dim]
                diag2 = result2.persistence_diagrams[dim]

                if len(diag1) > 0 and len(diag2) > 0:
                    # Filter out infinite persistence (essential features)
                    finite1 = diag1[np.isfinite(diag1).all(axis=1)]
                    finite2 = diag2[np.isfinite(diag2).all(axis=1)]

                    if len(finite1) == 0 or len(finite2) == 0:
                        continue

                    # Simple approximation: Wasserstein-1 distance
                    # (true bottleneck requires specialized library)
                    sorted1 = np.sort(finite1[:, 1] - finite1[:, 0])  # persistence
                    sorted2 = np.sort(finite2[:, 1] - finite2[:, 0])

                    # Pad shorter array
                    max_len = max(len(sorted1), len(sorted2))
                    if len(sorted1) < max_len:
                        sorted1 = np.pad(sorted1, (0, max_len - len(sorted1)), constant_values=0)
                    if len(sorted2) < max_len:
                        sorted2 = np.pad(sorted2, (0, max_len - len(sorted2)), constant_values=0)

                    distances.append(np.mean(np.abs(sorted1 - sorted2)))

        return float(np.mean(distances)) if distances else 0.0

    def detect_reasoning_loops(
        self,
        embeddings: np.ndarray,
        min_persistence: float = 0.2,
    ) -> list[TopologicalFeature]:
        """Detect reasoning loops (H1 features) in memory.

        Args:
            embeddings: Memory embeddings.
            min_persistence: Minimum persistence for a loop to be significant.

        Returns:
            List of detected reasoning loops.
        """
        result = self.analyze(embeddings)
        return [
            f
            for f in result.significant_features
            if f.dimension == 1 and f.persistence >= min_persistence
        ]

    def detect_knowledge_gaps(
        self,
        embeddings: np.ndarray,
        min_persistence: float = 0.3,
    ) -> list[TopologicalFeature]:
        """Detect knowledge gaps (H2 features) in memory.

        Args:
            embeddings: Memory embeddings.
            min_persistence: Minimum persistence for a gap to be significant.

        Returns:
            List of detected knowledge gaps.
        """
        result = self.analyze(embeddings)
        return [
            f
            for f in result.significant_features
            if f.dimension == 2 and f.persistence >= min_persistence
        ]

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _validate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Validate and convert embeddings to numpy array."""
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D array (n_memories, embedding_dim), "
                f"got shape {embeddings.shape}"
            )

        if embeddings.shape[0] == 0:
            raise ValueError("embeddings cannot be empty")

        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contain non-finite values (NaN or Inf)")

        return embeddings.astype(np.float64)

    def _find_representative_indices(
        self,
        dim: int,
        birth: float,
        death: float,
        dist_matrix: np.ndarray,
        embeddings: np.ndarray,
    ) -> list[int]:
        """Find memory indices most representative of a topological feature.

        Args:
            dim: Homology dimension (0=cluster, 1=loop, 2=void)
            birth: Birth scale of the feature
            death: Death scale of the feature
            dist_matrix: Pairwise distance matrix
            embeddings: Original embeddings

        Returns:
            List of memory indices involved in this feature
        """
        n_points = len(embeddings)

        if dim == 0:
            # H0: Connected component
            # Find points that merge at this birth scale
            threshold = birth * 1.1  # Small tolerance
            candidates = []

            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if dist_matrix[i, j] <= threshold:
                        candidates.extend([i, j])

            # Return unique indices, up to 5 representatives
            unique = list(dict.fromkeys(candidates))
            return unique[:5] if unique else [0]

        elif dim == 1:
            # H1: Loop/cycle
            # Find points that form a cycle at birth scale
            candidates = []
            for i in range(n_points):
                neighbors_in_range = 0
                for j in range(n_points):
                    if i != j and birth <= dist_matrix[i, j] <= death:
                        neighbors_in_range += 1

                # Points with many neighbors in the loop range are on the cycle
                if neighbors_in_range >= 2:
                    candidates.append(i)

            return candidates[:8] if candidates else [0]

        elif dim == 2:
            # H2: Void/cavity
            # Find points that border the void
            candidates = []
            for i in range(n_points):
                neighbors_outside = sum(
                    1 for j in range(n_points) if i != j and dist_matrix[i, j] > birth
                )
                # Points bordering void have many distant neighbors
                if neighbors_outside > n_points * 0.3:
                    candidates.append(i)

            return candidates[:6] if candidates else [0]

        else:
            # Higher dimensions — just return first few points
            return list(range(min(3, n_points)))

    def _compress_topological(
        self,
        result: TopologyResult,
        n_keep: int,
    ) -> list[int]:
        """Keep memories with highest topological significance."""
        if not result.significant_features:
            # No significant features — keep first n_keep
            return list(range(n_keep))

        # Score each memory by its involvement in significant features
        scores: dict[int, float] = {}
        for feature in result.significant_features:
            for idx in feature.indices:
                scores[idx] = scores.get(idx, 0.0) + feature.persistence

        # Sort by score (descending) and keep top n_keep
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return sorted_indices[:n_keep]

    def _compress_diversity(
        self,
        result: TopologyResult,
        n_keep: int,
    ) -> list[int]:
        """Keep one memory from each topological cluster (H0)."""
        if len(result.persistence_diagrams) == 0 or len(result.persistence_diagrams[0]) == 0:
            return list(range(n_keep))

        # Extract connected components from H0 diagram
        # This is a simplified approach — full implementation would use
        # the cocycles from ripser to get actual cluster assignments
        h0_diagram = result.persistence_diagrams[0]
        if len(h0_diagram) == 0:
            return list(range(n_keep))

        # For now, use a simple diversity sampling
        # Keep memories that are spread across the persistence diagram
        sorted_by_birth = np.argsort(h0_diagram[:, 0])
        step = max(1, len(sorted_by_birth) // n_keep)
        return [int(index) for index in sorted_by_birth[::step][:n_keep]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze agent memory topology using persistent homology",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with synthetic data",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2,
        help="Maximum homology dimension (default: 2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Persistence threshold (default: 0.1)",
    )
    args = parser.parse_args()

    if args.demo:
        print("Running demo with synthetic memory embeddings...\n")

        # Generate synthetic data: 3 clusters + 1 loop
        np.random.seed(42)
        n_per_cluster = 30

        # Cluster 1: concept A
        cluster1 = np.random.randn(n_per_cluster, 10) + np.array([5] + [0] * 9)

        # Cluster 2: concept B
        cluster2 = np.random.randn(n_per_cluster, 10) + np.array([0, 5] + [0] * 8)

        # Cluster 3: concept C
        cluster3 = np.random.randn(n_per_cluster, 10) + np.array([0, 0, 5] + [0] * 7)

        # Loop: reasoning cycle
        theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        loop = np.column_stack(
            [
                np.cos(theta) * 3,
                np.sin(theta) * 3,
                np.zeros(40),
            ]
            + [np.zeros(40)] * 7
        )

        embeddings = np.vstack([cluster1, cluster2, cluster3, loop])

        print(f"Generated {len(embeddings)} synthetic memories")
        print(f"  - 3 conceptual clusters ({n_per_cluster} memories each)")
        print("  - 1 reasoning loop (40 memories)")
        print()

        # Analyze
        analyzer = MemoryTopologyAnalyzer(
            max_dimension=args.max_dim,
            persistence_threshold=args.threshold,
        )
        result = analyzer.analyze(embeddings)
        print(result.summary())
        print()

        # Detect specific features
        loops = analyzer.detect_reasoning_loops(embeddings)
        print(f"Detected {len(loops)} reasoning loops")
        for i, loop in enumerate(loops[:3], 1):
            print(f"  {i}. persistence={loop.persistence:.3f}, indices={loop.indices}")
        print()

        gaps = analyzer.detect_knowledge_gaps(embeddings)
        print(f"Detected {len(gaps)} knowledge gaps")
        for i, gap in enumerate(gaps[:3], 1):
            print(f"  {i}. persistence={gap.persistence:.3f}, indices={gap.indices}")
        print()

        # Compress
        compressed, kept_indices = analyzer.compress(embeddings, keep_ratio=0.3)
        print(f"Compressed {len(embeddings)} → {len(compressed)} memories")
        print(f"Kept {len(kept_indices)} topologically significant memories")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
