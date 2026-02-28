# `gpu_backend.py` — Detailed Documentation

This module provides a **GPU-accelerated alternative** to the CPU-based (Numba) leaf similarity computations used elsewhere in the project (`leaf_pred_top_100_search.py`, `leaf_pred_class_stats.py`). It uses **CuPy** to manage GPU memory and launch a custom **CUDA kernel**.

---

## Core Concept: Leaf Similarity

The task is: given a set of **reference samples** and **query samples**, each described by their **leaf indices** across `n_trees` in a random forest, compute how many trees assign each query to the **same leaf** as each reference. That count is the "leaf similarity score."

---

## Module Structure

### 1. GPU Availability Check

```python
_CUPY_AVAILABLE = False
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    pass
```

A safe import guard. `is_gpu_available()` returns `True` only if CuPy is installed and importable. This lets callers gracefully fall back to CPU.

---

### 2. The CUDA Kernel

This is the performance-critical core — a raw CUDA C++ kernel string compiled at runtime by CuPy.

**Inputs:**

| Parameter | Shape | Description |
|---|---|---|
| `refs_T` | `(n_trees, n_refs_padded)` | Reference leaf indices, **transposed** for coalesced memory access |
| `queries` | `(batch_size, n_trees)` | Query leaf indices, row-major |
| `sims` | `(batch_size, n_refs_padded)` | Output similarity counts |

**Execution model:**

- **One CUDA block per query** (`blockIdx.x = query index`).
- Each block's threads cooperatively process all references for that one query.

**Step-by-step:**

1. **Shared memory load:** All threads cooperatively load the query's `n_trees` leaf indices into shared memory (`s_query[]`). This avoids redundant global memory reads — every thread in the block needs the same query vector.

2. **`__syncthreads()`:** Barrier to ensure shared memory is fully populated before any thread reads it.

3. **Vectorized comparison loop:**
   - `n_refs_padded` is guaranteed to be a multiple of 4, so references are reinterpreted as `int4` vectors (128-bit loads — 4 x `int32`).
   - Each thread handles a stride of `int4` elements (`vr += blockDim.x`).
   - For each tree `t`, it loads 4 reference leaf indices at once via `__ldg()` (read-only texture cache path — higher bandwidth than normal global loads on modern GPUs).
   - It compares each of the 4 reference leaves against the query's leaf for that tree (`qv == r4.x`, etc.), accumulating boolean results as integer counts (`c0-c3`).
   - After iterating all trees, the 4 counts are written back as a single `int4` store.

**Why this is fast:**

- **128-bit coalesced loads:** Adjacent threads read adjacent `int4` values, providing full memory bus utilization.
- **`__ldg()` cache hints:** Routes reads through the read-only texture cache, which has higher bandwidth than L1 for broadcast-heavy patterns.
- **Shared memory for queries:** The query vector is reused `n_refs/4/blockDim.x` times per thread — shared memory avoids repeated global reads.
- **`#pragma unroll 4`:** Hints the compiler to unroll the inner tree loop by 4 iterations, reducing loop overhead.

---

### 3. Helper Functions

- **`_compile_kernel()`** — Wraps `cp.RawKernel()`. CuPy caches the compiled PTX after the first call, so subsequent invocations are instant.
- **`_pad_to_multiple(n, m)`** — Rounds `n` up to the next multiple of `m` (used to ensure `n_refs` is a multiple of 4 for `int4` alignment).

---

### 4. Top-K Extraction: `gpu_topk()`

Extracts the top-`k` most similar references **per query**, entirely on GPU:

- If `k >= n_cols`: just `argsort` the whole row (degenerate case).
- Otherwise: uses `cp.argpartition` (O(n) radix-select) to find the top-k indices without fully sorting, then sorts only those k elements.
- Transfers only the small `(batch_size, k)` result to CPU — avoids moving the full similarity matrix.
- Returns `list[list[tuple[float, int]]]` matching the CPU code's format.

---

### 5. Full Similarity Transfer: `gpu_full_similarities()`

Simply copies the entire `(batch_size, n_refs)` similarity matrix to CPU as `float64` arrays — one per query. Used when downstream code needs all pairwise scores (e.g., for class-level statistics).

---

### 6. `GPUSimilarityEngine` Class

The main API. Manages the GPU-resident reference data and kernel dispatch.

**`__init__(self, all_leaves_np)`:**

1. Casts the reference matrix to `int32` (from `float64` pickle files).
2. **Pads** `n_refs` to a multiple of 4 with sentinel value `-1` (never matches any real leaf index, so padding contributes 0 to scores).
3. **Transposes** to `(n_trees, n_refs_padded)` for coalesced access in the kernel.
4. Uploads to GPU via `cp.asarray()`.
5. Compiles the CUDA kernel.
6. Sets block size to **512 threads** (good occupancy on Blackwell/Hopper/Ampere).

**`compute_batch(self, query_batch_np, mode, k)`:**

1. Uploads the query batch to GPU as `int32`.
2. Allocates a zeroed output buffer `(batch_size, n_refs_padded)`.
3. Launches the kernel with:
   - Grid: `(batch_size,)` — one block per query.
   - Block: `(512,)` threads.
   - Shared memory: `n_trees * 4` bytes.
4. **Trims padding** columns from the output (`[:, :n_refs]`).
5. Dispatches to `gpu_topk()` or `gpu_full_similarities()` based on `mode`.

---

## Data Flow Summary

```
all_leaves (n_refs, n_trees)
    -> pad to multiple of 4
    -> transpose to (n_trees, n_refs_padded)
    -> upload to GPU

query_batch (batch_size, n_trees)
    -> upload to GPU

CUDA kernel: for each query (1 block), for each tree,
    compare query leaf vs 4 ref leaves at a time (int4)
    -> sims (batch_size, n_refs_padded)

    -> trim padding
    -> topk extraction (GPU-side argpartition + argsort)
    -> small result transferred to CPU
```

---

## Usage

```python
from leaf_similarity.gpu_backend import is_gpu_available, GPUSimilarityEngine

if is_gpu_available():
    engine = GPUSimilarityEngine(all_leaves)
    results = engine.compute_batch(query_batch, mode="topk", k=100)
```

---

## Compatibility

- Requires **sm_75+** (Turing or newer — RTX 2000+, T4+, A100, H100, B100).
- The `int4` vectorized pattern works on all of these, but the `__ldg` hints and high thread counts are tuned for **high-bandwidth GPUs** (Ampere+).
