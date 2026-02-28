import argparse
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np

from leaf_similarity.leaf_pred_helper import load_leaf_dataset, read_metadata
from leaf_similarity.leaf_pred_top_100_search import get_similarities_for_target_leaves


def get_families_to_count(metadata):
    families = metadata.avclass.value_counts()
    return dict(families)


def cut_families_at_threshold(family_ty_count, threshold):
    return {k: v for k, v in family_ty_count.items() if v >= threshold}


def get_entry_family_by_index(metadata, index):
    return metadata.iloc[index].avclass


def get_match_count_percentage(
    percentage, metadata, families_to_count, len_train, similarities_vector, j
):
    index = len_train + j
    entry_class = get_entry_family_by_index(metadata, index)
    if entry_class not in families_to_count:
        return entry_class, -1
    target_percentage = int(percentage / 100 * families_to_count[entry_class])
    row = [(x[1], x[0]) for x in enumerate(similarities_vector)]
    top_hits = sorted(row, reverse=True)[:target_percentage]
    counter = 0
    for hit in top_hits:
        hit_class = get_entry_family_by_index(metadata, hit[1])
        if entry_class == hit_class:
            counter += 1
    return entry_class, counter / target_percentage * 100


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


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_families_percentage_similarities_statistics_test_vs_train_test(
    leaves_train_dataset_path,
    leaves_test_dataset_path,
    ember_metadata_path,
    output_path,
    target_percentage=10,
    minimum_count_threshold=100,
    use_gpu=False,
    batch_size=None,
):
    leaves_train = load_leaf_dataset(leaves_train_dataset_path)
    leaves_test = load_leaf_dataset(leaves_test_dataset_path)
    all_leaves = np.concatenate((leaves_train, leaves_test), axis=0)

    metadata = read_metadata(ember_metadata_path)
    families_to_count = get_families_to_count(metadata)
    families_to_count = cut_families_at_threshold(families_to_count, minimum_count_threshold)

    results = {c: [] for c in families_to_count}

    len_train = len(leaves_train)

    engine = _try_init_gpu_engine(all_leaves, use_gpu)

    if engine is not None:
        # GPU path
        batch_size = batch_size or 10_000
        n_queries = len(leaves_test)
        start = 0
        while start < n_queries:
            end = min(start + batch_size, n_queries)
            query_batch = leaves_test[start:end]
            similarities = engine.compute_batch(query_batch, mode="full")
            my_partial = partial(
                get_match_count_percentage, target_percentage, metadata, families_to_count, len_train
            )
            js = list(range(start, end))
            with Pool() as pool:
                parallel_results = pool.starmap(
                    my_partial, [(similarities[x - start], x) for x in js]
                )
            parallel_results = sorted(parallel_results, key=lambda x: x[0])
            for r in parallel_results:
                if r[1] == -1:
                    continue
                results[r[0]].append(r[1])
            print(f"[GPU] Done {end}/{n_queries}")
            start = end
    else:
        # CPU path (original logic)
        batch_size = batch_size or 1_000
        start = 0
        end = batch_size
        while end <= len(leaves_test):
            similarities = get_similarities_for_target_leaves(leaves_test, all_leaves, start, end)
            my_partial = partial(
                get_match_count_percentage, target_percentage, metadata, families_to_count, len_train
            )
            js = []
            simils = []
            for j in range(start, end):
                js.append(j)
                simils.append(similarities[j - start])
            with Pool() as pool:
                parallel_results = pool.starmap(
                    my_partial, [(simils[x], js[x]) for x in range(len(js))]
                )
            parallel_results = sorted(parallel_results, key=lambda x: x[0])
            for r in parallel_results:
                if r[1] == -1:
                    continue
                results[r[0]].append(r[1])
            start += batch_size
            end += batch_size

    with open(output_path, "w") as f:
        for c, counts in results.items():
            f.write(f"{c}, {np.mean(counts)}" + "\n")


def main(args):
    compute_families_percentage_similarities_statistics_test_vs_train_test(
        args.leaf_train_dataset_path,
        args.leaf_test_dataset_path,
        args.ember_metadata_path,
        args.output_path,
        args.target_percentage,
        args.minimum_count_threshold,
        use_gpu=args.gpu,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    description = (
        "Compute the statistics of retrieval per class having the train and test leaf datasets"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        default="",
        help="Output path for the result file.",
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
        "--ember_metadata_path",
        type=str,
        required=False,
        default="",
        help="Path to the ember metadata.",
    )
    parser.add_argument(
        "--target_percentage",
        type=int,
        required=False,
        default=10,
        help="What percentage of the samples from a class should be the target number for a sample in that specific class.",
    )
    parser.add_argument(
        "--minimum_count_threshold",
        type=int,
        required=False,
        default=100,
        help="Ignore the class that has less than the threshould number of entries.",
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
