import argparse
import pickle
import warnings
from multiprocessing import Pool

import numpy as np
from numba import njit, prange

from leaf_similarity.leaf_pred_helper import load_leaf_dataset


@njit(parallel=True)
def get_similarities_for_target_leaves(target_leaves, all_leaves, start, end):
    length_all_leaves = len(all_leaves)
    lenght_target_leaves = len(target_leaves)
    similarities = [np.zeros(length_all_leaves) for _ in range(end - start)]
    for i, leaves_one in enumerate(target_leaves):
        if i < start or i >= end:
            continue
        for j in prange(length_all_leaves):
            leaves_two = all_leaves[j]
            similarity = np.count_nonzero(leaves_one == leaves_two)
            similarities[i - start][j] = similarity
        if (i + 1) % 100 == 0:
            print(f"Done {i+1} entries from {lenght_target_leaves}..")
    return similarities


def get_top_similarities(i, similarities):
    row = [(x[1], x[0]) for x in enumerate(similarities)]
    return i, sorted(row, reverse=True)[:100]


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _try_init_gpu_engine(all_leaves, use_gpu):
    """Attempt to create a GPUSimilarityEngine; fall back to CPU on failure."""
    if not use_gpu:
        return None
    from leaf_similarity.gpu_backend import is_gpu_available, GPUSimilarityEngine

    if not is_gpu_available():
        warnings.warn(
            "CuPy is not installed — falling back to CPU. "
            "Install with: pip install cupy-cuda12x",
            stacklevel=2,
        )
        return None
    return GPUSimilarityEngine(all_leaves)


def _gpu_top100_batched(engine, target_leaves, output_folder_path, batch_size):
    """Run the full top-100 search on GPU in batches, writing pickle shards."""
    n_queries = len(target_leaves)
    start = 0
    while start < n_queries:
        end = min(start + batch_size, n_queries)
        query_batch = target_leaves[start:end]
        topk = engine.compute_batch(query_batch, mode="topk", k=100)
        results = [(i, topk[i]) for i in range(len(topk))]
        with open(f"{output_folder_path}/{start}.pkl", "wb") as g:
            pickle.dump(results, g)
        print(f"[GPU] Done {end}/{n_queries}")
        start = end


# ---------------------------------------------------------------------------
# CPU path (original logic, unchanged)
# ---------------------------------------------------------------------------

def _cpu_top100_batched(target_leaves, all_leaves, output_folder_path, batch_size):
    """Run the full top-100 search on CPU using the Numba kernel."""
    n_queries = len(target_leaves)
    start = 0
    end = batch_size
    while end <= n_queries:
        similarities = get_similarities_for_target_leaves(target_leaves, all_leaves, start, end)
        with Pool() as pool:
            results = pool.starmap(
                get_top_similarities, [(i, x) for i, x in enumerate(similarities)]
            )
        results = sorted(results, key=lambda x: x[0])
        with open(f"{output_folder_path}/{start}.pkl", "wb") as g:
            pickle.dump(results, g)
        start += batch_size
        end += batch_size
        print(f"Done {start}..")


# ---------------------------------------------------------------------------
# Top-level functions
# ---------------------------------------------------------------------------

def get_top_100_similar_test_vs_train_test(
    leaves_train_dataset_path, leaves_test_dataset_path, output_folder_path,
    use_gpu=False, batch_size=None,
):
    leaves_train = load_leaf_dataset(leaves_train_dataset_path)
    leaves_test = load_leaf_dataset(leaves_test_dataset_path)
    all_leaves = np.concatenate((leaves_train, leaves_test), axis=0)

    engine = _try_init_gpu_engine(all_leaves, use_gpu)
    if engine is not None:
        _gpu_top100_batched(engine, leaves_test, output_folder_path, batch_size or 10_000)
    else:
        _cpu_top100_batched(leaves_test, all_leaves, output_folder_path, batch_size or 1_000)


def get_top_100_similarities_unlabelled_vs_train_test(
    leaves_train_dataset_path,
    leaves_test_dataset_path,
    leaves_unlabelled_dataset_path,
    output_folder_path,
    use_gpu=False,
    batch_size=None,
):
    leaves_train = load_leaf_dataset(leaves_train_dataset_path)
    leaves_test = load_leaf_dataset(leaves_test_dataset_path)
    leaves_unlabelled = load_leaf_dataset(leaves_unlabelled_dataset_path)
    all_leaves = np.concatenate((leaves_train, leaves_test), axis=0)

    engine = _try_init_gpu_engine(all_leaves, use_gpu)
    if engine is not None:
        _gpu_top100_batched(engine, leaves_unlabelled, output_folder_path, batch_size or 10_000)
    else:
        _cpu_top100_batched(leaves_unlabelled, all_leaves, output_folder_path, batch_size or 1_000)


def get_top_100_similarities_unlabelled_vs_train(
    leaves_train_dataset_path, leaves_unlabelled_dataset_path, output_folder_path,
    use_gpu=False, batch_size=None,
):
    leaves_train = load_leaf_dataset(leaves_train_dataset_path)
    leaves_unlabelled = load_leaf_dataset(leaves_unlabelled_dataset_path)

    engine = _try_init_gpu_engine(leaves_train, use_gpu)
    if engine is not None:
        _gpu_top100_batched(engine, leaves_unlabelled, output_folder_path, batch_size or 10_000)
    else:
        _cpu_top100_batched(leaves_unlabelled, leaves_train, output_folder_path, batch_size or 1_000)


def main(args):
    if args.command == "test_vs_train_test":
        get_top_100_similar_test_vs_train_test(
            args.leaf_train_dataset_path, args.leaf_test_dataset_path, args.output_folder_path,
            use_gpu=args.gpu, batch_size=args.batch_size,
        )
    elif args.command == "unlabelled_vs_train_test":
        get_top_100_similarities_unlabelled_vs_train_test(
            args.leaf_train_dataset_path,
            args.leaf_test_dataset_path,
            args.leaf_unlabelled_dataset_path,
            args.output_folder_path,
            use_gpu=args.gpu, batch_size=args.batch_size,
        )
    elif args.command == "unlabelled_vs_train":
        get_top_100_similarities_unlabelled_vs_train(
            args.leaf_train_dataset_path, args.leaf_unlabelled_dataset_path, args.output_folder_path,
            use_gpu=args.gpu, batch_size=args.batch_size,
        )
    else:
        print(
            "Command not supported. Choose from: test_vs_train_test; \
                unlabelled_vs_train_test; unlabelled_vs_train"
        )


if __name__ == "__main__":
    description = "Having two leaf predictions datasets A and B compute the top 100 similar entries from B for each sample in A."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--command",
        type=str,
        required=True,
        help="What datasets to compare; supported: test_vs_train_test; \
            unlabelled_vs_train_test; unlabelled_vs_train;",
    )
    parser.add_argument(
        "-o",
        "--output_folder_path",
        type=str,
        required=True,
        default="",
        help="Output path for the results; they will be saved in multiple pickle files.",
    )
    parser.add_argument(
        "--leaf_train_dataset_path",
        type=str,
        required=False,
        default="",
        help="Path to the leaf predictions train dataset.",
    )
    parser.add_argument(
        "--leaf_test_dataset_path",
        type=str,
        required=False,
        default="",
        help="Path to the leaf predictions test dataset.",
    )
    parser.add_argument(
        "--leaf_unlabelled_dataset_path",
        type=str,
        required=False,
        default="",
        help="Path to the leaf predictions unlabelled dataset.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration via CuPy (requires cupy-cuda12x).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of queries per batch. Defaults to 10000 (GPU) or 1000 (CPU).",
    )
    args = parser.parse_args()
    main(args)
