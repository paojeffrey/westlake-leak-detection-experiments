# Westlake Leak Detection — Two-Stage Experiments

Databricks notebooks for finding liquid leaks in plant inspection photos. The ANYbotics robot photographs the same points of interest (POIs) again and again, so detection works the way a person would: **first notice that a spot looks different from its normal history, then decide what the change actually is.**

| Stage | Notebook | What it does |
|-------|----------|--------------|
| **1. Change detection** | `poi_baseline_experiments.py` | Compares each image to the same POI's normal (no-leak) history. Six candidate models are evaluated against the current production model. The winner (**Model D**, an unsupervised ensemble of POI-relative image features and DINOv2 PatchCore-lite) is registered to Unity Catalog. |
| **2. AI verification** | `stage2_llm_verification.py` | Each image flagged by Model D is sent to a vision LLM, together with a known-dry reference image of the same POI and the stage 1 change heatmap. The LLM decides whether the change is a liquid leak or something else (an object, lighting, camera angle, an old stain…). Four LLMs are compared. |

**Result:** Model D + **Gemini 3.8 Flash** catches **5 of 6 leaks with 0 false alarms**. The current production model catches 3 of 6 with 49 false alarms. See [Stage 2 results](#stage-2-results-oct-16-2026).

> [!IMPORTANT]
> **Update these names before running in Westlake's environment.**
> Every catalog, schema, table, volume, model and endpoint name in these notebooks points at the Databricks FE demo workspace (`stable_classic_wg38i9_catalog`). Westlake's workspace will use **different names**, and the notebooks will fail on the first data cell until you change them.
>
> | What | Variable / location | Notebook(s) | Current (demo) value | Change to |
> |------|---------------------|-------------|----------------------|-----------|
> | MLflow experiment | `EXPERIMENT` (cell *Setup and Data Loading*) | both | `/Users/jeffrey.pao@databricks.com/Leak Detection - POI Baseline Experiments` | A workspace path you own |
> | Labeled source table | `SPLITS_TABLE` (cell *Setup and Data Loading*) | both | `stable_classic_wg38i9_catalog.westlake_gas_leakage_2.silver_liquid_leak_results_splits` | Westlake labeled results table |
> | Image volume folder | `IMAGE_BASE_PATH` (cell *Setup and Data Loading*) | both | `/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/OneDrive_1_10-13-2026/` | Westlake volume + folder |
> | Registered model | `MODEL_NAME` (cell *Define Model D Pyfunc, Log and Register as Champion*) | stage 1 | `stable_classic_wg38i9_catalog.westlake_gas_leakage_2.leak_detection_model` | `<westlake_catalog>.<westlake_schema>.<model_name>` |
> | LLM serving endpoints | `LLMS` (cell *LLM Client and Response Parsing*) | stage 2 | `databricks-gemini-3-8-flash`, `databricks-claude-sonnet-5-5`, `databricks-claude-opus-4-7`, `databricks-gpt-6-1-sol` | Endpoints available in Westlake's workspace (keep at least Gemini 3.8 Flash) |
>
> The source table must also have the columns listed in [Input data](#input-data).

## Files

| File | What it is |
|------|------------|
| `notebooks/poi_baseline_experiments.py` | Databricks source format. Best for diffs and for importing back into a workspace. |
| `notebooks/poi_baseline_experiments.ipynb` | Jupyter format **with outputs** from the last run (Oct 18, 2026), including the PatchCore heatmaps and result tables. Viewable directly on GitHub. |
| `notebooks/stage2_llm_verification.py` | Stage 2 (Databricks source format). The results below come from its MLflow runs. |

These are exports of the workspace notebooks `/Users/jeffrey.pao@databricks.com/Leak Detection - POI Baseline Experiments (notebook)` and `.../Leak Detection - Stage 2 LLM Verification (notebook)`. Both log to the same MLflow experiment.

# Stage 1: Change detection

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

# Stage 2: AI verification

Stage 1 is tuned to catch as many leaks as possible, which means it also produces false alarms (new objects, lighting changes, camera shifts). Stage 2 sends each stage 1 alert to a vision LLM to remove them. **Stage 2 can only remove alerts, so its recall is capped by stage 1.**

### How it works

1. **Stage 1 candidates:** the notebook recomputes Model D exactly as in the baseline notebook. Candidate sets come from normal-image score percentiles only (no tuning on leaks):
   - **10% budget:** the threshold where 10% of normal images alert (stage 1: 4/6 leaks, 15 false alarms)
   - **20% budget:** a wider net, so stage 2 has a chance to keep 5/6 leaks (stage 1: 5/6 leaks, 31 false alarms)
2. **Reference image:** the most similar known-dry image of the same POI from a *different day*, chosen by DINOv2 CLS similarity. POIs with no history get no reference.
3. **What the LLM sees:** the reference image, the current image, and the current image with the stage 1 change heatmap overlaid. Without a reference, it sees only the current image and the heatmap.
4. **Prompt:** lists liquid leak indicators (a new wet or glossy patch, pooling, drips, a wet area below a flange or valve) and common false-alarm causes (new or moved objects, lighting and shadows, camera angle, old stains also visible in the reference, people). The model must return JSON: `leak`, `change_type`, `confidence`, `evidence`. The exact prompts are logged to MLflow as `prompts.json`.
5. **Endpoints:** called through Databricks Foundation Model APIs using the OpenAI-compatible client (`/serving-endpoints`).

### Stage 2 results (Oct 16, 2026)

Out of 156 images (6 leaks, 150 normal):

| System | Budget | Leaks caught | False alarms | Precision | F1 | LLM calls | p50 latency |
|---|---|---|---|---|---|---|---|
| Current production model | — | 3/6 | 49 | 0.06 | 0.10 | — | — |
| Stage 1 only (Model D) | 10% | 4/6 | 15 | 0.21 | 0.32 | 0 | — |
| Stage 1 only (Model D) | 20% | 5/6 | 31 | 0.14 | 0.24 | 0 | — |
| **D + Gemini 3.8 Flash** | **20%** | **5/6** | **0** | **1.00** | **0.909** | 36 | 2.7 s |
| D + Gemini 3.8 Flash | 10% | 4/6 | 0 | 1.00 | 0.800 | 19 | 2.8 s |
| D + Claude Sonnet 5.5 | 20% | 4/6 | 0 | 1.00 | 0.800 | 36 | 2.3 s |
| D + Claude Sonnet 5.5 | 10% | 3/6 | 0 | 1.00 | 0.667 | 19 | 2.5 s |
| D + GPT-6.1 Sol | 20% | 4/6 | 1 | 0.80 | 0.727 | 36 | 4.4 s |
| D + GPT-6.1 Sol | 10% | 3/6 | 0 | 1.00 | 0.667 | 19 | 3.6 s |
| D + Claude Opus 4.7 | 20% | 3/6 | 2 | 0.60 | 0.545 | 36 | 2.6 s |
| D + Claude Opus 4.7 | 10% | 2/6 | 2 | 0.50 | 0.400 | 19 | 2.6 s |

**Takeaways**
- **Gemini 3.8 Flash removed all 31 stage 1 false alarms and kept all 5 leaks stage 1 caught.** The one missed leak was never flagged by stage 1.
- Widening the stage 1 net from 10% to 20% only pays off with a verifier this accurate. Gemini turns the extra false alarms into an extra caught leak with no added noise.
- Verification costs about 36 LLM calls (≈2.7 s each) per 156 images, because only stage 1 candidates are sent.
- The Gemini 3.8 Flash 20% run (`0e8469945c044789adf1f26624f19b92`) is tagged as the experiment's `champion_run_id`. It also produced the `leak_alerts` table used by the [Leak Alert Review](https://github.com/paojeffrey/leak-alert-review) app.

### MLflow artifacts per stage 2 run

- `candidate_verdicts.json`: every candidate with its stage 1 score, reference file, LLM verdict, change type, confidence and evidence
- `prompts.json`: the exact prompts, with and without a reference image
- The summary run (`Summary - Stage 2 LLM verification`) logs `stage2_summary.json` (the table above) and `stage2_all_verdicts.json`

# Running the notebooks

1. Import the notebooks from `notebooks/` into a Databricks workspace.
2. Update the names in the table at the top of this README.
3. Attach to serverless (environment version 6) or a cluster with internet access, so it can download `facebook/dinov2-small` from Hugging Face.
4. **Stage 1:** run all cells through *Comparison Summary*. Then run the two Model D cells manually if you want to register the model.
5. **Stage 2:** run all cells. It recomputes stage 1 itself, so it doesn't depend on the stage 1 notebook having been run. It needs access to the Foundation Model API endpoints listed in `LLMS`; remove any that aren't available.
