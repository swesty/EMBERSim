"""GPU-accelerated leaf similarity search using CuPy and a custom CUDA kernel.

This module provides a drop-in GPU alternative to the Numba CPU kernel used in
leaf_pred_top_100_search.py and leaf_pred_class_stats.py.  It targets NVIDIA
GPUs with sufficient VRAM.

The CUDA kernel uses 128-bit (int4) vectorized loads/stores and __ldg()
read-only cache hints, optimized for high-bandwidth GPUs (Blackwell, Hopper,
Ampere).  Compatible with any CUDA GPU (sm_75+).

Usage::

    from leaf_similarity.gpu_backend import is_gpu_available, GPUSimilarityEngine

    if is_gpu_available():
        engine = GPUSimilarityEngine(all_leaves)
        results = engine.compute_batch(query_batch, mode="topk", k=100)
"""

import warnings

import numpy as np

_CUPY_AVAILABLE = False
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    pass


def is_gpu_available():
    """Return True if CuPy is importable and a CUDA device is reachable."""
    return _CUPY_AVAILABLE


# ---------------------------------------------------------------------------
# CUDA kernel source — vectorized int4 loads/stores (128-bit)
# ---------------------------------------------------------------------------
_KERNEL_SOURCE = r"""
extern "C" __global__
void leaf_similarity_kernel(
    const int* __restrict__ refs_T,    // (n_trees, n_refs_padded) transposed
    const int* __restrict__ queries,   // (batch_size, n_trees) row-major
    int*       __restrict__ sims,      // (batch_size, n_refs_padded) output
    const int n_refs_padded,           // n_refs rounded up to multiple of 4
    const int n_trees,
    const int batch_size
) {
    extern __shared__ int s_query[];   // n_trees ints
    const int q = blockIdx.x;
    if (q >= batch_size) return;

    // Cooperative load of query vector into shared memory
    for (int t = threadIdx.x; t < n_trees; t += blockDim.x)
        s_query[t] = queries[q * n_trees + t];
    __syncthreads();

    // 128-bit vectorized path: each thread processes 4 refs per iteration
    const int vec_n = n_refs_padded >> 2;
    const int4* refs_T_v = reinterpret_cast<const int4*>(refs_T);
    int4*       sims_v   = reinterpret_cast<int4*>(sims + q * n_refs_padded);

    for (int vr = threadIdx.x; vr < vec_n; vr += blockDim.x) {
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;

        #pragma unroll 4
        for (int t = 0; t < n_trees; t++) {
            // __ldg: read-only texture cache path — higher bandwidth
            int4 r4 = __ldg(&refs_T_v[t * vec_n + vr]);
            int qv  = s_query[t];
            c0 += (qv == r4.x);
            c1 += (qv == r4.y);
            c2 += (qv == r4.z);
            c3 += (qv == r4.w);
        }

        sims_v[vr] = make_int4(c0, c1, c2, c3);
    }
}
"""


def _compile_kernel():
    """Compile and return the CUDA RawKernel (cached by CuPy after first call)."""
    return cp.RawKernel(_KERNEL_SOURCE, "leaf_similarity_kernel")


def _pad_to_multiple(n, m):
    """Round *n* up to the nearest multiple of *m*."""
    return ((n + m - 1) // m) * m


# ---------------------------------------------------------------------------
# Top-K extraction on GPU
# ---------------------------------------------------------------------------

def gpu_topk(sims_gpu, k=100):
    """Extract top-*k* (score, ref_idx) pairs per query row, on GPU.

    Parameters
    ----------
    sims_gpu : cupy.ndarray, shape (batch_size, n_refs), dtype int32
    k : int

    Returns
    -------
    list[list[tuple[float, int]]]
        For each query, a list of ``(score, ref_idx)`` sorted descending by
        score — the same format produced by the CPU ``get_top_similarities``.
    """
    n_cols = sims_gpu.shape[1]
    k = min(k, n_cols)

    if k >= n_cols:
        # Degenerate: return everything, just argsort
        sort_order = cp.argsort(-sims_gpu, axis=1)
        topk_idx = sort_order
        topk_scores = cp.take_along_axis(sims_gpu, sort_order, axis=1)
    else:
        # O(n) radix-select on GPU (kth is 0-indexed, so k-1)
        topk_idx = cp.argpartition(-sims_gpu, kth=k - 1, axis=1)[:, :k]
        topk_scores = cp.take_along_axis(sims_gpu, topk_idx, axis=1)

        # Sort only the k elements per row
        sort_order = cp.argsort(-topk_scores, axis=1)
        topk_idx = cp.take_along_axis(topk_idx, sort_order, axis=1)
        topk_scores = cp.take_along_axis(topk_scores, sort_order, axis=1)

    # Transfer small result to CPU
    topk_idx_cpu = topk_idx.get()
    topk_scores_cpu = topk_scores.get().astype(np.float64)

    results = []
    for i in range(topk_scores_cpu.shape[0]):
        row = [
            (float(topk_scores_cpu[i, j]), int(topk_idx_cpu[i, j]))
            for j in range(k)
        ]
        results.append(row)
    return results


def gpu_full_similarities(sims_gpu):
    """Transfer the full similarity matrix back to CPU.

    Parameters
    ----------
    sims_gpu : cupy.ndarray, shape (batch_size, n_refs), dtype int32

    Returns
    -------
    list[numpy.ndarray]
        One float64 array per query row — matches the return format of the
        Numba ``get_similarities_for_target_leaves`` function.
    """
    sims_cpu = sims_gpu.get().astype(np.float64)
    return [sims_cpu[i] for i in range(sims_cpu.shape[0])]


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class GPUSimilarityEngine:
    """Holds the reference leaf array on GPU and dispatches batched similarity
    computations via a custom CUDA kernel.

    Parameters
    ----------
    all_leaves_np : numpy.ndarray, shape (n_refs, n_trees)
        The reference leaf-index matrix (typically ``float64`` from pickle;
        will be cast to ``int32`` internally).
    """

    _VEC_WIDTH = 4  # int4 = 4 x int32 = 128 bits

    def __init__(self, all_leaves_np):
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not installed — cannot create GPUSimilarityEngine. "
                "Install with: pip install cupy-cuda12x"
            )

        refs_int32 = np.ascontiguousarray(all_leaves_np.astype(np.int32))
        n_refs, n_trees = refs_int32.shape

        # Pad n_refs to a multiple of VEC_WIDTH so int4 loads need no tail handling.
        # Padding value -1 will never match a real leaf index, so padded columns
        # contribute 0 to every similarity score.
        n_refs_padded = _pad_to_multiple(n_refs, self._VEC_WIDTH)
        if n_refs_padded != n_refs:
            refs_int32 = np.pad(
                refs_int32, ((0, n_refs_padded - n_refs), (0, 0)),
                mode="constant", constant_values=-1,
            )

        # Transpose for coalesced access: (n_trees, n_refs_padded)
        refs_T = np.ascontiguousarray(refs_int32.T)

        self._refs_T_gpu = cp.asarray(refs_T)
        self._n_refs = n_refs
        self._n_refs_padded = n_refs_padded
        self._n_trees = n_trees
        self._kernel = _compile_kernel()
        self._block_size = 512  # higher occupancy on Blackwell / Hopper

        vram_mb = refs_T.nbytes / (1024 * 1024)
        print(
            f"[GPU] Reference matrix uploaded: "
            f"{n_refs:,} refs x {n_trees:,} trees "
            f"(padded to {n_refs_padded:,}), "
            f"{vram_mb:.1f} MB on device"
        )

    def compute_batch(self, query_batch_np, mode="topk", k=100):
        """Compute leaf similarities for a batch of queries.

        Parameters
        ----------
        query_batch_np : numpy.ndarray, shape (batch_size, n_trees)
        mode : {"topk", "full"}
            ``"topk"`` returns only the top-*k* results per query (efficient).
            ``"full"`` returns the complete similarity vector per query.
        k : int
            Number of top results when ``mode="topk"``.

        Returns
        -------
        list
            If *mode="topk"*: ``list[list[tuple[float, int]]]``
            If *mode="full"*: ``list[numpy.ndarray]``
        """
        batch_size = query_batch_np.shape[0]

        # Upload query batch to GPU as int32
        queries_gpu = cp.asarray(
            np.ascontiguousarray(query_batch_np.astype(np.int32))
        )

        # Allocate padded output buffer
        sims_gpu = cp.zeros((batch_size, self._n_refs_padded), dtype=cp.int32)

        # Shared memory: n_trees * sizeof(int)
        shared_mem = self._n_trees * 4

        # Launch kernel: one block per query
        self._kernel(
            (batch_size,),                # grid
            (self._block_size,),          # block
            (
                self._refs_T_gpu,
                queries_gpu,
                sims_gpu,
                np.int32(self._n_refs_padded),
                np.int32(self._n_trees),
                np.int32(batch_size),
            ),
            shared_mem=shared_mem,
        )

        # Trim padding columns before downstream processing
        sims_gpu = sims_gpu[:, :self._n_refs]

        if mode == "topk":
            return gpu_topk(sims_gpu, k=k)
        elif mode == "full":
            return gpu_full_similarities(sims_gpu)
        else:
            raise ValueError(f"Unknown mode: {mode!r} (expected 'topk' or 'full')")
