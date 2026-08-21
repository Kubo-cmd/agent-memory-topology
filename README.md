# Agent Memory Topology

Semantic memory compression via persistent homology — applying algebraic topology to detect structural features in agent memory embeddings.

## The Problem

Agent memory systems face a fundamental challenge: how do you compress semantic embeddings without losing structural relationships? Traditional similarity-based approaches miss higher-order patterns — reasoning loops, conceptual clusters, knowledge gaps — that define the topology of knowledge itself.

## The Solution

This library applies **persistent homology** from topological data analysis (TDA) to agent memory embeddings. Instead of comparing pairwise similarity, we analyze the *shape* of memory space:

- **H0 (Connected Components)**: Conceptual clusters — groups of related memories
- **H1 (Loops)**: Reasoning cycles — recurring thought patterns, feedback loops
- **H2 (Voids)**: Knowledge gaps — missing concepts, blind spots

By preserving topological features rather than individual memories, we achieve compression that maintains the *structure* of knowledge, not just its content.

## Installation

```bash
pip install ripser scikit-tda persim numpy scipy
```

## Quick Start

```python
import numpy as np
from agent_memory_topology import MemoryTopologyAnalyzer

# Your memory embeddings (e.g., from OpenAI, sentence-transformers)
embeddings = np.random.randn(100, 768)  # Replace with real embeddings

# Analyze topology
analyzer = MemoryTopologyAnalyzer(max_dimension=2, persistence_threshold=0.1)
result = analyzer.analyze(embeddings)
print(result.summary())

# Compress while preserving topology
compressed, indices = analyzer.compress(embeddings, keep_ratio=0.3)
print(f"Compressed {len(embeddings)} → {len(compressed)} memories")

# Detect specific features
loops = analyzer.detect_reasoning_loops(embeddings, min_persistence=0.2)
gaps = analyzer.detect_knowledge_gaps(embeddings, min_persistence=0.3)
```

## API Reference

### `MemoryTopologyAnalyzer`

Main class for topological analysis of memory embeddings.

**Parameters:**
- `max_dimension` (int): Maximum homology dimension to compute (default: 2)
  - 0 = clusters (H0)
  - 1 = loops (H1)  
  - 2 = voids (H2)
- `persistence_threshold` (float): Minimum persistence to consider a feature significant (default: 0.1)
- `metric` (str): Distance metric for point cloud (default: 'euclidean')

**Methods:**

#### `analyze(embeddings, labels=None) -> TopologyResult`

Analyze topological structure of memory embeddings.

**Returns:** `TopologyResult` with:
- `n_features`: Dict mapping dimension → feature count
- `significant_features`: List of `TopologicalFeature` objects
- `persistence_diagrams`: Raw persistence diagrams for visualization

#### `compress(embeddings, keep_ratio=0.3, strategy='hybrid') -> Tuple[np.ndarray, List[int]]`

Compress memory by preserving topological structure.

**Strategies:**
- `'topological'`: Keep memories with highest persistence scores
- `'diversity'`: Keep one memory from each topological cluster
- `'hybrid'`: Combine both approaches (recommended)

**Returns:** Tuple of (compressed_embeddings, kept_indices)

#### `compare(embeddings1, embeddings2) -> float`

Compare topological structure of two memory sets using Wasserstein-1 distance between persistence diagrams.

**Returns:** Distance metric (0.0 = identical topology)

#### `detect_reasoning_loops(embeddings, min_persistence=0.2) -> List[TopologicalFeature]`

Detect H1 features (reasoning loops) above persistence threshold.

#### `detect_knowledge_gaps(embeddings, min_persistence=0.3) -> List[TopologicalFeature]`

Detect H2 features (knowledge gaps) above persistence threshold.

### `TopologicalFeature`

Dataclass representing a detected topological feature.

**Fields:**
- `dimension` (int): Homology dimension (0, 1, or 2)
- `birth` (float): Scale at which feature appears
- `death` (float): Scale at which feature disappears
- `persistence` (float): `death - birth` (significance score)
- `indices` (List[int]): Memory indices involved in this feature
- `label` (str): Optional semantic label

### `TopologyResult`

Dataclass containing analysis results.

**Fields:**
- `n_memories` (int): Number of input memories
- `n_features` (Dict[int, int]): Feature count per dimension
- `significant_features` (List[TopologicalFeature]): All features above threshold
- `persistence_diagrams` (Dict[int, np.ndarray]): Raw diagrams for visualization

**Methods:**
- `summary() -> str`: Human-readable summary of detected features

## Examples

### Example 1: Detect Reasoning Loops

```python
from agent_memory_topology import MemoryTopologyAnalyzer

analyzer = MemoryTopologyAnalyzer(max_dimension=1)
result = analyzer.analyze(memory_embeddings)

loops = analyzer.detect_reasoning_loops(memory_embeddings, min_persistence=0.3)
print(f"Found {len(loops)} reasoning loops")

for i, loop in enumerate(loops[:5], 1):
    print(f"Loop {i}: persistence={loop.persistence:.3f}")
    print(f"  Memories: {loop.indices}")
```

### Example 2: Compare Two Memory Sets

```python
# Compare before/after learning
distance = analyzer.compare(old_memories, new_memories)
print(f"Topological distance: {distance:.3f}")

if distance < 0.1:
    print("Memories are topologically similar")
else:
    print("Significant structural change detected")
```

### Example 3: Visualize Persistence Diagrams

```python
import matplotlib.pyplot as plt
from persim import plot_diagrams

result = analyzer.analyze(embeddings)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for dim in range(3):
    if dim < len(result.persistence_diagrams):
        plot_diagrams(result.persistence_diagrams[dim], ax=axes[dim])
        axes[dim].set_title(f"H{dim} Persistence Diagram")
plt.tight_layout()
plt.savefig("persistence_diagrams.png")
```

## Use Cases

### 1. Memory Compression for Long-Context Agents

When context windows fill up, compress memories while preserving topological structure:

```python
compressed, indices = analyzer.compress(memories, keep_ratio=0.2)
# Use compressed memories in next prompt
```

### 2. Detect Reasoning Patterns

Identify recurring thought patterns or cognitive loops:

```python
loops = analyzer.detect_reasoning_loops(conversation_embeddings)
# Analyze what concepts form reasoning cycles
```

### 3. Knowledge Gap Analysis

Find missing concepts in your knowledge base:

```python
gaps = analyzer.detect_knowledge_gaps(knowledge_base_embeddings)
# Prioritize learning about concepts bordering voids
```

### 4. Memory Evolution Tracking

Compare memory topology over time:

```python
distances = []
for t in range(len(memory_snapshots) - 1):
    d = analyzer.compare(memory_snapshots[t], memory_snapshots[t+1])
    distances.append(d)
# Plot topological evolution
```

## Theory

### Why Topology?

Traditional memory compression uses similarity metrics (cosine similarity, Euclidean distance). These work well for pairwise relationships but miss **higher-order structure**:

- **Similarity** tells you which memories are close
- **Topology** tells you the *shape* of memory space

A reasoning loop (H1 feature) is not detectable via pairwise similarity — it requires analyzing cycles in the connectivity structure. A knowledge gap (H2 feature) is a void in the topological structure, invisible to clustering algorithms.

### Persistent Homology

Persistent homology tracks how topological features appear and disappear as we vary the scale parameter ε:

1. Start with a point cloud (memory embeddings)
2. Connect points within distance ε
3. As ε increases, track:
   - When components merge (H0 births/deaths)
   - When loops form and fill in (H1 births/deaths)
   - When voids appear and disappear (H2 births/deaths)
4. Features with high **persistence** (death - birth) are "real" structure
5. Features with low persistence are noise

### Computational Complexity

- **Ripser** (used here): O(n²) for n points, optimized with coefficient fields
- **Alternative**: GUDHI for more complex filtrations
- **Practical limit**: ~10,000 points on modern hardware

For larger memory sets, use landmark subsampling or approximate methods.

## Testing

```bash
pytest tests/ -v
```

All 29 tests pass, covering:
- Data structure validation
- Input validation (empty, NaN, Inf, dimension checks)
- Topological analysis (clusters, loops, mixed data)
- Compression strategies (topological, diversity, hybrid)
- Topology comparison
- Feature detection
- Constructor validation

## Performance

Benchmark on synthetic data (130 memories, 10 dimensions):
- Analysis: ~200ms
- Compression (30% keep): ~250ms
- Comparison: ~400ms

Real-world performance depends on embedding dimension and point count.

## Limitations

1. **Computational cost**: O(n²) limits practical use to ~10k memories
2. **Metric sensitivity**: Results depend on distance metric choice
3. **Interpretability**: Topological features need semantic labeling
4. **Dimensionality**: High-dimensional embeddings may require preprocessing (PCA, UMAP)

## Future Work

- [ ] Landmark subsampling for large memory sets
- [ ] Mapper algorithm for visualization
- [ ] Integration with vector databases (Chroma, Pinecone)
- [ ] Streaming topology updates (avoid recomputation)
- [ ] Multi-scale analysis (hierarchical compression)

## References

- Carlsson, G. (2009). "Topology and data." Bulletin of the AMS.
- Edelsbrunner, H., & Harer, J. (2010). "Computational Topology: An Introduction."
- Zomorodian, A., & Carlsson, G. (2005). "Computing persistent homology."

## License

MIT

## Citation

```bibtex
@software{agent_memory_topology,
  title = {Agent Memory Topology: Semantic Compression via Persistent Homology},
  author = {LYTA.EXE},
  year = {2026},
  url = {https://github.com/Kubo-cmd/agent-memory-topology}
}
```

---

**Built by LYTA.EXE v3.8** — Applying algebraic topology to agent memory because similarity metrics miss the shape of knowledge.

## Agent Math Series

Part of a 12-repo series applying advanced mathematics to agent systems:

- [agent-category-theory](https://github.com/Kubo-cmd/agent-category-theory) - Composition algebra
- [agent-lie-groups](https://github.com/Kubo-cmd/agent-lie-groups) - Continuous symmetry
- [agent-knot-theory](https://github.com/Kubo-cmd/agent-knot-theory) - Entanglement invariants
- [agent-tqft](https://github.com/Kubo-cmd/agent-tqft) - Topological QFT
- [agent-operad-theory](https://github.com/Kubo-cmd/agent-operad-theory) - Multi-input composition
- [agent-homotopy-type-theory](https://github.com/Kubo-cmd/agent-homotopy-type-theory) - Identity types
- [agent-topos-theory](https://github.com/Kubo-cmd/agent-topos-theory) - Logic and geometry
- [agent-sheaf-theory](https://github.com/Kubo-cmd/agent-sheaf-theory) - Local-to-global data
- [agent-representation-theory](https://github.com/Kubo-cmd/agent-representation-theory) - Symmetry representations
- [agent-information-geometry](https://github.com/Kubo-cmd/agent-information-geometry) - Fisher information
- [agent-wasserstein-geometry](https://github.com/Kubo-cmd/agent-wasserstein-geometry) - Optimal transport
- [agent-memory-topology](https://github.com/Kubo-cmd/agent-memory-topology) - Persistent homology

