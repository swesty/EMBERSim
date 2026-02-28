# Inference: Querying New PE Files Against the Reference Database

This document describes how to take a new, unseen PE file and find high-probability matches in the reference database using leaf similarity.

---

## Prerequisites (One-Time Setup)

Before you can query anything, you need three artifacts built from the EMBER dataset:

1. **Trained XGBoost model** (`model.json`) — trained via `xgboost_trainer.py` with `n_estimators=2048`
2. **Reference leaf pickles** — run `leaf_pred_predictions.py` with `pred_leaf=True` on the train and test sets to get `leaves_train.pkl` and `leaves_test.pkl` (each is a numpy array of shape `(n_samples, 2048)`)
3. **EMBER metadata CSV** — maps sample indices to SHA256 hashes, labels (0/1), and AVClass family tags

---

## Inference Flow for a New PE File

### Step 1: Feature Extraction

```python
import ember

# EMBER + LIEF parse the PE file and produce a 2381-dim feature vector
features = ember.features("suspicious_file.exe")
# -> numpy array, shape (2381,)
```

### Step 2: Get Leaf Indices

```python
import xgboost as xgb

model = xgb.Booster()
model.load_model("model.json")

dmat = xgb.DMatrix(features.reshape(1, -1))  # shape (1, 2381)
query_leaves = model.predict(dmat, pred_leaf=True)
# -> shape (1, 2048) -- one leaf index per tree
```

This produces a vector of 2,048 integers -- the leaf node each tree assigned the sample to. This vector is the sample's **structural fingerprint**.

### Step 3: Load the Reference Database

```python
import pickle, numpy as np

with open("leaves_train.pkl", "rb") as f:
    leaves_train = pickle.load(f)
with open("leaves_test.pkl", "rb") as f:
    leaves_test = pickle.load(f)

all_leaves = np.concatenate((leaves_train, leaves_test), axis=0)
# -> shape (n_refs, 2048)
```

### Step 4: Compute Similarities

For each reference sample, count how many of the 2,048 trees assign the query and the reference to the same leaf:

```python
similarities = np.count_nonzero(query_leaves == all_leaves, axis=1)
# -> shape (n_refs,) -- one score per reference, range [0, 2048]
```

### Step 5: Get Top Matches

```python
top_k = 100
top_indices = np.argsort(-similarities)[:top_k]
top_scores = similarities[top_indices]

# Map indices back to metadata
import pandas as pd
metadata = pd.read_csv("ember_metadata.csv")
metadata = metadata[metadata["label"] != -1].reset_index(drop=True)

for idx, score in zip(top_indices, top_scores):
    row = metadata.iloc[idx]
    print(f"Score: {score}/2048  SHA256: {row['sha256']}  "
          f"Label: {'malware' if row['label']==1 else 'benign'}  "
          f"Family: {row['avclass']}")
```

### Step 6: Interpret Results

A score of **2048/2048** means every single tree agreed -- nearly identical structure. In practice:

- **>1900**: Very high confidence match -- likely same malware family or variant
- **1500-1900**: Strong structural similarity
- **<1000**: Weak match, probably unrelated

The majority label among the top-K matches gives you a binary prediction (malware/benign), and the most common AVClass family tag gives you a family classification.

---

## At Scale (GPU)

For batch queries (thousands of files), the GPU backend avoids the O(n_queries x n_refs) bottleneck:

```python
from leaf_similarity.gpu_backend import GPUSimilarityEngine

engine = GPUSimilarityEngine(all_leaves)  # uploads refs to VRAM once
results = engine.compute_batch(query_leaves, mode="topk", k=100)
```

See [GPU_BACKEND.md](GPU_BACKEND.md) for details on the CUDA kernel and GPU engine.

---

## Pipeline Diagram

```
New PE file
  -> EMBER feature extraction (2,381-dim vector)
  -> model.predict(X, pred_leaf=True)  ->  leaf indices [2048 ints]
  -> similarity search vs reference DB  ->  top matches + scores
  -> majority vote / family voting      ->  prediction + confidence
  -> output: label, similar SHA256s, scores
```

---

## File Dependency Chain

```
xgboost_trainer.py          -> trained model (JSON)
leaf_pred_predictions.py    -> leaf index pickles (train/test/unlabeled)
leaf_pred_top_100_search.py -> top-100 similarity pickles (CPU or GPU)
leaf_pred_binary_stats.py   -> binary classification stats
leaf_pred_class_stats.py    -> multi-class family stats
leaf_pred_top_100_shas.py   -> human-readable CSV results
e2e.py / evaluation.py      -> Relevance@K evaluation metrics
```

---

## Why Leaf Indices Instead of Probabilities?

Leaf indices capture the **complete decision path** through all 2,048 trees -- a much richer representation than a single probability. Two files that land in the same leaves across most trees are structurally similar according to the entire ensemble, making this approach particularly effective at finding malware variants and families.
