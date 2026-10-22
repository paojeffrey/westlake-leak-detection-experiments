# Westlake Leak Detection — POI Baseline Experiments

A Databricks notebook that tests whether liquid leaks in plant inspection photos can be found by comparing each image to **the same point of interest's (POI's) normal history**, instead of judging each image on its own. The ANYbotics robot photographs the same POIs again and again, so a leak shows up as a change from what that spot usually looks like.

The notebook trains and evaluates six candidate models plus the current production baseline, logs every run to MLflow, and registers the winner (**Model D**, an unsupervised ensemble) to Unity Catalog.

> [!IMPORTANT]
> **Update these names before running in Westlake's environment.**
> Every catalog, schema, table, volume and model name in this notebook points at the Databricks FE demo workspace (`stable_classic_wg38i9_catalog`). Westlake's workspace will use **different names**, and the notebook will fail on the first data cell until you change them.
>
> | What | Variable / location in the notebook | Current (demo) value | Change to |
> |------|-------------------------------------|----------------------|-----------|
> | MLflow experiment | `EXPERIMENT` (cell *Setup and Data Loading*) | `/Users/jeffrey.pao@databricks.com/Leak Detection - POI Baseline Experiments` | A workspace path you own |
> | Labeled source table | `SPLITS_TABLE` (cell *Setup and Data Loading*) | `stable_classic_wg38i9_catalog.westlake_gas_leakage_2.silver_liquid_leak_results_splits` | Westlake labeled results table |
> | Image volume folder | `IMAGE_BASE_PATH` (cell *Setup and Data Loading*) | `/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/OneDrive_1_10-13-2026/` | Westlake volume + folder |
> | Registered model | `MODEL_NAME` (cell *Define Model D Pyfunc, Log and Register as Champion*) | `stable_classic_wg38i9_catalog.westlake_gas_leakage_2.leak_detection_model` | `<westlake_catalog>.<westlake_schema>.<model_name>` |
>
> The source table must also have the columns listed in [Input data](#input-data).

## Files

| File | What it is |
|------|------------|
| `notebooks/poi_baseline_experiments.py` | Databricks source format. Best for diffs and for importing back into a workspace. |
| `notebooks/poi_baseline_experiments.ipynb` | Jupyter format **with outputs** from the last run (Oct 18, 2026), including the PatchCore heatmaps and result tables. Viewable directly on GitHub. |

Both are exports of the workspace notebook `/Users/jeffrey.pao@databricks.com/Leak Detection - POI Baseline Experiments (notebook)`.

## Models compared

| Model | Idea | Needs leak examples to train? |
|---|---|---|
| Current production model | `leak_suspected` column from the table | — |
| A. Absolute features + LR | 10 hand-crafted image features, logistic regression (prior best approach) | Yes |
| B1. POI-relative feature anomaly | RMS of robust z-scores vs. this POI's normal images | **No** |
| B2. POI-relative features + LR | Logistic regression on the POI-relative z-scores | Yes |
| C1. DINOv2 global kNN | 1 − cosine similarity of the DINOv2 CLS embedding to the nearest normal image of the same POI | **No** |
| C2. DINOv2 PatchCore-lite | Patch-level nearest-neighbor distance to the POI's normal patch bank (also shows *where* the change is) | **No** |
| D. Ensemble B1 + C2 | Average percentile rank of B1 and C2 | **No** |

### Evaluation protocol

- 156 images with a matching file in the volume: 6 leaks, 150 normal, 13 POIs, 12 days.
- **Leave-one-day-out**: each image is scored using only images from *other days*, so images of the same leak event never inform each other.
- POI reference = labeled-normal images of the same POI from other days (minimum 5). If there are fewer, it falls back to normal images from all POIs (4 images, at POIs 358, 398, 399 and 400).
- Primary metrics: **false alarms needed to catch k of 6 leaks**, and leaks caught under a fixed false-alarm budget (5% / 10% of normal images). Budget thresholds come from normal-image score percentiles only. No thresholds are tuned on leak images.

## Results (last run, Oct 18, 2026)

| Model | ROC-AUC | PR-AUC | FP to catch 6/6 | FP to catch 4/6 | Caught @ 5% budget | Caught @ 10% budget |
|---|---|---|---|---|---|---|
| Current production model | — | — | — | — | 3/6 (49 FP, binary alert) | 3/6 (49 FP, binary alert) |
| A. Absolute features + LR | 0.374 | 0.072 | 149 | 147 | 1/6 (8 FP) | 1/6 (15 FP) |
| B1. POI-relative feature anomaly | 0.799 | 0.163 | 106 | 22 | 1/6 (8 FP) | 3/6 (15 FP) |
| B2. POI-relative features + LR | 0.310 | 0.049 | 148 | 148 | 0/6 (8 FP) | 1/6 (15 FP) |
| C1. DINOv2 global kNN | 0.646 | 0.105 | 132 | 34 | 1/6 (8 FP) | 2/6 (15 FP) |
| C2. DINOv2 PatchCore-lite | **0.853** | 0.165 | **68** | 16 | 1/6 (8 FP) | 3/6 (15 FP) |
| **D. Ensemble B1 + C2** | 0.840 | **0.180** | 89 | **12** | **2/6 (8 FP)** | **4/6 (15 FP)** |

**Takeaways**
- Comparing against each POI's own history works much better than absolute features. A and B2 (the supervised models) do worse than random with only 6 leaks to learn from.
- Model D catches 4 of 6 leaks with 15 false alarms. The current production model catches 3 of 6 with 49.
- The best models are fully unsupervised, so they don't depend on collecting more leak examples.

## Notebook walkthrough

| Cell | What it does |
|------|--------------|
| Install Dependencies | `mlflow scikit-learn scipy matplotlib pillow torch torchvision transformers` |
| Setup and Data Loading | Loads the labeled table, keeps rows whose image exists in the volume, and derives `day`, `poi`, `y` and `current_alert` |
| Reference Selection and Evaluation Helpers | `reference_idx()` (same-POI normals with fallback), `evaluate()`, and `log_model()` (logs to MLflow) |
| Current Production Model Baseline | Scores the existing `leak_suspected` alert |
| Feature Extraction | 10 features from the bottom 60% (floor area) of each image |
| POI-Relative Feature Transform | Robust z-scores (median / MAD) vs. the POI's normal reference |
| A, B1, B2 | Feature-based models |
| DINOv2 Embeddings | `facebook/dinov2-small` CLS and patch tokens at ~448px |
| C1, C2 | Embedding-based models. C2 also saves `patchcore_heatmaps.png` as an MLflow artifact |
| D | Rank-average ensemble of B1 + C2 |
| Comparison Summary | Side-by-side table, alerts per POI, and a summary MLflow run. **Ends with `dbutils.notebook.exit(...)`** |
| Prepare Reference Artifacts for Model D | Pickles the per-POI normal features / embeddings / patches, C2 medians, historical scores, config, and DINOv2 weights |
| Define Model D Pyfunc, Log and Register as Champion | `LeakDetectionEnsemble` MLflow pyfunc. Registers it to UC and sets the `Champion` alias |

> [!NOTE]
> The *Comparison Summary* cell calls `dbutils.notebook.exit()`, so **Run all** (or a job run) stops there and the last two registration cells don't run. To register Model D, run those two cells manually after the summary, as was done for the Oct 18 run.

## Input data

The source table (`SPLITS_TABLE`) needs these columns:

| Column | Used for |
|--------|----------|
| `filename` | Matched against files in `IMAGE_BASE_PATH` |
| `point_of_interest_id` | POI grouping (cast to `int`) |
| `created_on_date_time` | Day used for leave-one-day-out splits |
| `leaked` | Ground truth (`1.0` = leak) |
| `leak_suspected` | Current production alert (`"Yes"` / other) |
| `split` | Shown in the leak preview table |

## Registered model (Model D)

- **Input:** DataFrame with `image_path` (str) and `poi_id` (int)
- **Output:** `anomaly_score` (0–1, average percentile rank), `b1_score`, `c2_score`, `b1_rank`, `c2_rank`
- POIs with fewer than 5 normal reference images fall back to the pooled reference from all POIs.

In the demo workspace, this notebook produced **v2** of `leak_detection_model` (MLflow run `db63beefecbf48ca8eabaedfef3285a0`). That version has since been **superseded by v3** (run `f2cc26eac4fe4b1293c15e54eb13a8fe`), which was built from v2's artifacts by a separate notebook, `Leak Detection - Register Model D v3`, and now holds the `Champion` alias.

## Running it

1. Import `notebooks/poi_baseline_experiments.py` (or the `.ipynb`) into a Databricks workspace.
2. Update the names in the table at the top of this README.
3. Attach to serverless (environment version 6) or a cluster with internet access, so it can download `facebook/dinov2-small` from Hugging Face.
4. Run all cells through *Comparison Summary*, then run the two Model D cells manually if you want to register the model.
