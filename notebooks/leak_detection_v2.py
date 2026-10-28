# Databricks notebook source
# DBTITLE 1,Notebook Title
# MAGIC %md
# MAGIC # Leak Detection v2 — Aligned Change Detection + Vision-LLM Verification
# MAGIC
# MAGIC One notebook, two stages, the way a person would inspect: **first notice that a spot looks different from its own
# MAGIC history, then decide what the change actually is.**
# MAGIC
# MAGIC **Stage 1 — change detection (plain Python + DINOv2, no LLM).** Each image is compared with dry images of the same
# MAGIC point of interest (POI) from other days:
# MAGIC
# MAGIC | Model | Idea |
# MAGIC |---|---|
# MAGIC | E1-448 / E1-896 | **Aligned PatchCore**: estimate the camera shift to each dry reference, compare each patch only with the same location (±small window), then z-score each location against how much it varies across the POI's dry images. Score = mean of top 0.1% z. 896 = 2×2 tiles at 448 (2× resolution). |
# MAGIC | E1-896 raw | Ablation: aligned distances without per-location normalization |
# MAGIC | E2 | **New dark patch**: pixel-level L* darkening vs. the aligned dry references, robust to small shifts and lighting. Score = area of largest darkened blob. |
# MAGIC | **F** | Rank-average of E1-896 + E2 (the v2 stage 1 model) |
# MAGIC | G | Rank-average of B1 + C2 + E1-896 + E2 |
# MAGIC | B1, C2, D | v1 baselines, recomputed for comparison |
# MAGIC
# MAGIC **Stage 2 — AI verification.** Images where the `STAGE1_MODEL` score is in the top 10% / 20% of normal-image scores
# MAGIC are sent to each vision LLM in `LLMS`, with a dry reference image, the change heatmap and **zoomed, camera-aligned
# MAGIC crops** around the strongest hotspots. Each image is checked `REPEATS` times and the majority verdict wins.
# MAGIC
# MAGIC **POIs without history** (fewer than 5 dry images of the same POI) cannot be change-detected. They skip stage 1
# MAGIC ("new POI → always verify") and the LLM judges the image on its own.
# MAGIC
# MAGIC Protocol: leave-one-day-out, alert thresholds from normal-image scores only, leak labels used only for grading.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install -q openai mlflow scikit-learn scipy matplotlib pillow torch torchvision transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup and Data Loading
import os
import io
import re
import time
import json
import base64
import numpy as np
import pandas as pd
import mlflow
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score, average_precision_score
from concurrent.futures import ThreadPoolExecutor
from matplotlib import cm as mpl_cm

EXPERIMENT = "/Users/jeffrey.pao@databricks.com/Leak Detection - POI Baseline Experiments"
mlflow.set_experiment(EXPERIMENT)

SPLITS_TABLE = "stable_classic_wg38i9_catalog.westlake_gas_leakage_2.silver_liquid_leak_results_splits"
IMAGE_BASE_PATH = "/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/OneDrive_1_10-13-2026/"
# Cached features and embeddings, so reruns skip the slow image pass
CACHE = "/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/experiments_cache/stage1_v2/"
os.makedirs(CACHE, exist_ok=True)

# Stage 2 settings
dbutils.widgets.text("STAGE1_MODEL", "F. Ensemble E1-896 + E2")
dbutils.widgets.text("LLMS", "Gemini 3.8 Flash=databricks-gemini-3-8-flash,Claude Sonnet 5.5=databricks-claude-sonnet-5-5")
dbutils.widgets.text("REPEATS", "3")
STAGE1_MODEL = dbutils.widgets.get("STAGE1_MODEL")
LLMS = [dict(zip(("name", "endpoint"), x.split("="))) for x in dbutils.widgets.get("LLMS").split(",")]
REPEATS = int(dbutils.widgets.get("REPEATS"))
MIN_REF = 5
RNG = np.random.default_rng(42)

df = spark.table(SPLITS_TABLE).toPandas()
volume_files = set(os.listdir(IMAGE_BASE_PATH))
df = df[df["filename"].isin(volume_files)].drop_duplicates("filename").sort_values("filename").reset_index(drop=True)
df["ts"] = pd.to_datetime(df["created_on_date_time"])
df["day"] = df["ts"].dt.strftime("%Y-%m-%d")
df["poi"] = df["point_of_interest_id"].astype(int)
df["y"] = (df["leaked"] == 1.0).astype(int)
df["current_alert"] = (df["leak_suspected"] == "Yes").astype(int)

y = df["y"].values
N_LEAKS = int(y.sum())
print(f"{len(df)} images | {N_LEAKS} leaks | {int((y == 0).sum())} normal | {df['poi'].nunique()} POIs | {df['day'].nunique()} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1 — Change detection

# COMMAND ----------

# DBTITLE 1,Reference Selection and Evaluation Helpers
poi_arr = df["poi"].values
day_arr = df["day"].values
normal_arr = y == 0


def reference_idx(i, exclude_days=()):
    """Normal images of the same POI from other days; fall back to all POIs if too few."""
    base = normal_arr & (day_arr != day_arr[i]) & ~np.isin(day_arr, list(exclude_days))
    same = base & (poi_arr == poi_arr[i])
    if same.sum() >= MIN_REF:
        return np.where(same)[0], False
    return np.where(base)[0], True


no_history = np.array([reference_idx(i)[1] for i in range(len(df))])
print(f"No-history images (routed to stage 2 unconditionally): {int(no_history.sum())} "
      f"(leaks: {int(y[no_history].sum())}, POIs: {sorted(set(poi_arr[no_history].tolist()))})")


def fp_at_recall(scores, y, k):
    leak_scores = np.sort(scores[y == 1])[::-1]
    if k > len(leak_scores) or not np.isfinite(leak_scores[k - 1]):
        return np.nan
    return int(((scores >= leak_scores[k - 1]) & (y == 0)).sum())


def at_threshold(scores, y, thr):
    pred = scores >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, alerts=tp + fp, precision=prec, recall=rec, f1=f1)


def evaluate(scores, y):
    scores = np.asarray(scores, dtype=float)
    m = {"roc_auc": roc_auc_score(y, scores), "pr_auc": average_precision_score(y, scores)}
    for k in (6, 5, 4, 3):
        m[f"fp_to_catch_{k}_of_6"] = fp_at_recall(scores, y, k)
    # Fixed false-alarm budgets, set from normal-image scores only (no leak labels used)
    for budget in (5, 10, 20):
        thr = np.percentile(scores[y == 0], 100 - budget)
        for k, v in at_threshold(scores, y, thr).items():
            m[f"budget{budget}_{k}"] = v
    order = (-scores).argsort(kind="stable").argsort() + 1
    m["leak_ranks"] = sorted(int(r) for r in order[y == 1])
    return m


RESULTS = {}
SCORES = pd.DataFrame({"filename": df["filename"], "poi": df["poi"], "day": df["day"], "leaked": y,
                       "no_history": no_history})


def log_model(name, scores, params):
    m = evaluate(scores, y)
    RESULTS[name] = m
    SCORES[name] = scores
    with mlflow.start_run(run_name=f"v2 {name}"):
        mlflow.log_params({**params, "num_images": len(df), "num_leaks": N_LEAKS,
                           "eval_protocol": "leave-one-day-out", "min_poi_reference": MIN_REF})
        mlflow.log_metrics({k: float(v) for k, v in m.items()
                            if k != "leak_ranks" and v is not None and np.isfinite(v)})
        mlflow.log_dict({"leak_ranks": m["leak_ranks"]}, "leak_ranks.json")
    print(f"{name}: ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} | FP to catch 6/6={m['fp_to_catch_6_of_6']} "
          f"5/6={m['fp_to_catch_5_of_6']} | 10%: {m['budget10_tp']}/6 ({m['budget10_fp']} FP) | "
          f"20%: {m['budget20_tp']}/6 ({m['budget20_fp']} FP) | leak ranks={m['leak_ranks']}")
    return m


def pct_rank(s):
    return pd.Series(s).rank(pct=True).values


def route_no_history(s):
    """POIs without history can't be change-detected: always send them to stage 2 (top of the ranking)."""
    s = np.asarray(s, dtype=float).copy()
    s[no_history] = np.nanmax(s[~no_history]) + 1.0
    return s

# COMMAND ----------

# DBTITLE 1,Load Images: Features (B1), L* at 960x540, DINOv2 Embeddings at 448 and 896
import torch
from transformers import AutoModel

torch.set_grad_enabled(False)
dino = AutoModel.from_pretrained("facebook/dinov2-small").eval()
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
SMALL = (960, 540)  # pixel-level work (alignment and E2) at 1/4 of 4K


def lab_l(rgb):
    r, g, b = [rgb[:, :, c].astype(np.float32) / 255 for c in range(3)]
    yy = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.where(yy > 0.008856, 116 * np.cbrt(yy) - 16, 903.3 * yy).astype(np.float32)


def rgb_to_sv(rgb):
    r, g, b = [rgb[:, :, c].astype(float) for c in range(3)]
    maxc = np.maximum(np.maximum(r, g), b); minc = np.minimum(np.minimum(r, g), b)
    return np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-9), 0), maxc


def extract_features(img):
    """Same 10 floor-area features as the baseline notebook (input to B1)."""
    w, h = img.size
    floor = np.array(img.crop((0, int(h * 0.4), w, h)))
    gray = floor.astype(float).mean(axis=2)
    s_ch, v_ch = rgb_to_sv(floor)
    dark = (gray < max(np.median(gray) * 0.75, 20)).astype(np.int32)
    lab, n = ndimage.label(dark)
    sizes = ndimage.sum(dark, lab, range(1, n + 1)) if n else []
    lap = ndimage.laplace(gray)
    lm = ndimage.uniform_filter(gray, 15); lsq = ndimage.uniform_filter(gray ** 2, 15)
    edges = np.hypot(ndimage.sobel(gray, 1), ndimage.sobel(gray, 0))
    emask = edges > np.percentile(edges, 85)
    bmask = ndimage.binary_dilation(dark > 0, iterations=5) & ~(dark > 0)
    L = lab_l(floor)
    return [float(v_ch.mean()), float(v_ch.std()), float(dark.mean()),
            int(sum(1 for s in sizes if s > dark.size * 0.002)), float(np.var(lap)), float(np.mean(lsq - lm ** 2)),
            float(emask.mean()), float((emask & bmask).sum() / max(bmask.sum(), 1)),
            float(np.mean(np.abs(L - np.median(L)))), float(s_ch[dark > 0].mean()) if dark.sum() else 0.0]


def embed(img, short, tiles):
    """DINOv2 patch tokens on a (tiles x tiles) grid of 448-px-ish crops, reassembled into one patch grid.
    Returns (cls of the full image if tiles == 1 else None, patches [H, W, 384] float16)."""
    w, h = img.size
    s = short / min(w, h)
    nw = round(w * s / 14 / tiles) * 14 * tiles
    nh = round(h * s / 14 / tiles) * 14 * tiles
    arr = (np.asarray(img.resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255 - MEAN) / STD
    th, tw = nh // tiles, nw // tiles
    crops = [arr[a * th:(a + 1) * th, b * tw:(b + 1) * tw] for a in range(tiles) for b in range(tiles)]
    x = torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2).copy()).float()
    out = dino(pixel_values=x).last_hidden_state
    pt = torch.nn.functional.normalize(out[:, 1:], dim=-1).numpy()
    gh, gw = th // 14, tw // 14
    grid = np.zeros((nh // 14, nw // 14, pt.shape[-1]), dtype=np.float16)
    for t, (a, b) in enumerate([(a, b) for a in range(tiles) for b in range(tiles)]):
        grid[a * gh:(a + 1) * gh, b * gw:(b + 1) * gw] = pt[t].reshape(gh, gw, -1)
    cls = torch.nn.functional.normalize(out[0, 0], dim=0).numpy() if tiles == 1 else None
    return cls, grid


cache_files = {k: os.path.join(CACHE, f"{k}.npy") for k in ("feat", "lstar", "cls", "p448", "p896")}
cached_names = os.path.join(CACHE, "filenames.json")
if all(os.path.exists(p) for p in cache_files.values()) and json.load(open(cached_names)) == df["filename"].tolist():
    F = np.load(cache_files["feat"]); LSTAR = np.load(cache_files["lstar"]); CLS = np.load(cache_files["cls"])
    P448 = np.load(cache_files["p448"]); P896 = np.load(cache_files["p896"])
    print("Loaded cached features and embeddings")
else:
    F, LSTAR, CLS, P448, P896 = [], [], [], [], []
    t0 = time.time()
    for n, f in enumerate(df["filename"]):
        img = Image.open(os.path.join(IMAGE_BASE_PATH, f)).convert("RGB")
        F.append(extract_features(img))
        LSTAR.append(lab_l(np.asarray(img.resize(SMALL, Image.BILINEAR))))
        cls, p = embed(img, 448, 1); CLS.append(cls); P448.append(p)
        _, p = embed(img, 896, 2); P896.append(p)
        if n % 20 == 0:
            print(f"  {n}/{len(df)} images, {time.time() - t0:.0f}s")
    F = np.array(F, dtype=float); LSTAR = np.stack(LSTAR); CLS = np.stack(CLS)
    P448 = np.stack(P448); P896 = np.stack(P896)
    for k, v in zip(("feat", "lstar", "cls", "p448", "p896"), (F, LSTAR, CLS, P448, P896)):
        np.save(cache_files[k], v)
    json.dump(df["filename"].tolist(), open(cached_names, "w"))
    print(f"Computed and cached in {time.time() - t0:.0f}s")
print(f"P448 grid {P448.shape[1:3]}, P896 grid {P896.shape[1:3]}, L* {LSTAR.shape[1:]}")

# COMMAND ----------

# DBTITLE 1,Baseline: B1, C2 (PatchCore-lite) and D, as in the baseline notebook
def poi_relative(F):
    Z = np.zeros_like(F)
    for i in range(len(F)):
        ref, _ = reference_idx(i)
        R = F[ref]
        med = np.median(R, axis=0)
        mad = 1.4826 * np.median(np.abs(R - med), axis=0)
        Z[i] = (F[i] - med) / np.where(mad > 1e-9, mad, R.std(axis=0) + 1e-9)
    return np.clip(Z, -10, 10)


s_B1 = np.sqrt(np.mean(poi_relative(F) ** 2, axis=1))


def normalize_by_reference(raw):
    return np.array([raw[i] / (np.median(raw[reference_idx(i)[0]]) + 1e-9) for i in range(len(raw))])


PFLAT = P448.reshape(len(df), -1, P448.shape[-1])
raw_C2 = np.zeros(len(df))
for i in range(len(df)):
    ref, _ = reference_idx(i)
    bank = PFLAT[ref].reshape(-1, PFLAT.shape[-1]).astype(np.float32)
    if len(bank) > 30000:
        bank = bank[RNG.choice(len(bank), 30000, replace=False)]
    q = torch.from_numpy(PFLAT[i].astype(np.float32)); b = torch.from_numpy(bank)
    d = torch.full((len(q),), float("inf"))
    for s in range(0, len(b), 10000):
        d = torch.minimum(d, torch.cdist(q, b[s:s + 10000]).min(dim=1).values)
    d = d.numpy()
    raw_C2[i] = float(np.sort(d)[-max(1, int(len(d) * 0.01)):].mean())
s_C2 = normalize_by_reference(raw_C2)
s_D = (pct_rank(s_B1) + pct_rank(s_C2)) / 2

log_model("B1. POI-relative feature anomaly", s_B1, {"approach": "baseline recomputed"})
log_model("C2. DINOv2 PatchCore-lite", s_C2, {"approach": "baseline recomputed"})
log_model("D. Ensemble B1 + C2", s_D, {"approach": "baseline recomputed"})
log_model("D + no-history routing", route_no_history(s_D), {"approach": "D, no-history POIs always verified"})

# COMMAND ----------

# DBTITLE 1,Camera Alignment (phase correlation between images of the same POI)
def shift_to(a, dy, dx):
    """out[r, c] = a[r + dy, c + dx] (NaN where that falls outside a)."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    H, W = a.shape[:2]
    r0, r1 = max(0, -dy), min(H, H - dy); c0, c1 = max(0, -dx), min(W, W - dx)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = a[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
    return out


WIN = np.outer(np.hanning(SMALL[1]), np.hanning(SMALL[0])).astype(np.float32)


def phase_shift(a, b):
    """Integer (dy, dx) such that a[r, c] ≈ b[r + dy, c + dx]."""
    Fa = np.fft.rfft2((a - a.mean()) * WIN); Fb = np.fft.rfft2((b - b.mean()) * WIN)
    R = Fa * np.conj(Fb)
    r = np.fft.irfft2(R / (np.abs(R) + 1e-9), s=a.shape)
    py, px = np.unravel_index(np.argmax(r), r.shape)
    py = py - a.shape[0] if py > a.shape[0] // 2 else py
    px = px - a.shape[1] if px > a.shape[1] // 2 else px
    return -int(py), -int(px)


# Sanity check of the sign convention on a synthetic shift
_t = shift_to(LSTAR[0], 7, -12)
_dy, _dx = phase_shift(np.nan_to_num(_t, nan=float(np.nanmean(_t))), LSTAR[0])
assert (_dy, _dx) == (7, -12), (_dy, _dx)

# Shift for every ordered pair of images of the same POI
SHIFT = {}
for poi in np.unique(poi_arr):
    idx = np.where(poi_arr == poi)[0]
    for i in idx:
        for j in idx:
            if i != j:
                SHIFT[(i, j)] = phase_shift(LSTAR[i], LSTAR[j])
mags = np.array([np.hypot(*v) for v in SHIFT.values()])
print(f"{len(SHIFT)} same-POI pairs | camera shift at 960x540: median {np.median(mags):.1f}px, "
      f"p90 {np.percentile(mags, 90):.1f}px, max {mags.max():.1f}px")

# COMMAND ----------

# DBTITLE 1,E1. Aligned PatchCore with Per-Location Normalization
def aligned_map(Q, P, dy, dx, R):
    """Per-patch distance from Q to the nearest patch of P within ±R of the aligned location (inf if none)."""
    H, W, _ = Q.shape
    best = torch.full((H, W), float("inf"))
    for oy in range(dy - R, dy + R + 1):
        for ox in range(dx - R, dx + R + 1):
            r0, r1 = max(0, -oy), min(H, H - oy); c0, c1 = max(0, -ox), min(W, W - ox)
            if r1 <= r0 or c1 <= c0:
                continue
            sim = (Q[r0:r1, c0:c1] * P[r0 + oy:r1 + oy, c0 + ox:c1 + ox]).sum(-1)
            best[r0:r1, c0:c1] = torch.minimum(best[r0:r1, c0:c1], (2 - 2 * sim).clamp_min(0))
    return best.sqrt().numpy()


def pair_maps(P, R):
    """Aligned distance map for every ordered same-POI pair, with the camera shift converted to patch units."""
    H, W = P.shape[1:3]
    sy, sx = H / SMALL[1], W / SMALL[0]
    T = [torch.from_numpy(p.astype(np.float32)) for p in P]
    return {(i, j): aligned_map(T[i], T[j], round(dy * sy), round(dx * sx), R)
            for (i, j), (dy, dx) in SHIFT.items()}


def e1_scores(A, grid, topk_frac=0.001, normalize=True):
    """Score each image against its POI references.

    Query map  = min over references j of A[i, j].
    Normalizer = per-location median / MAD of each reference's own leave-one-out map
                 (other references from other days), so locations that always vary (windows, lights) count less."""
    n_cells = grid[0] * grid[1]
    k = max(2, int(round(n_cells * topk_frac)))
    scores, zmaps = np.zeros(len(df)), {}
    for i in range(len(df)):
        ref, fb = reference_idx(i)
        if fb:
            scores[i] = np.nan
            continue
        q = np.min([A[(i, j)] for j in ref], axis=0)
        loo = np.stack([np.min([A[(r, j)] for j in ref if j != r and day_arr[j] != day_arr[r]], axis=0)
                        for r in ref if (day_arr[ref] != day_arr[r]).any()])
        loo[~np.isfinite(loo)] = np.nan; q = np.where(np.isfinite(q), q, np.nan)
        if normalize:
            mu = np.nanmedian(loo, axis=0)
            sd = ndimage.gaussian_filter(np.nan_to_num(1.4826 * np.nanmedian(np.abs(loo - mu), axis=0),
                                                       nan=np.nanmedian(loo)), 1.0)
            sd = np.maximum(sd, 0.5 * np.median(sd))
            z = (q - mu) / sd
        else:
            z = q / np.nanmedian(loo)
        zmaps[i] = z.astype(np.float32)
        v = z[np.isfinite(z)]
        scores[i] = float(np.sort(v)[-k:].mean())
    return scores, zmaps


t0 = time.time()
A448 = pair_maps(P448, R=1)
A896 = pair_maps(P896, R=2)
print(f"Aligned pair maps computed in {time.time() - t0:.0f}s")

s_E1_448, _ = e1_scores(A448, P448.shape[1:3])
s_E1_896, Z896 = e1_scores(A896, P896.shape[1:3])
s_E1_896_raw, _ = e1_scores(A896, P896.shape[1:3], normalize=False)
log_model("E1-448. Aligned PatchCore, location-normalized", route_no_history(s_E1_448),
          {"approach": "aligned PatchCore + per-location z", "resolution": 448, "window": 1, "topk_frac": 0.001})
log_model("E1-896. Aligned PatchCore, location-normalized", route_no_history(s_E1_896),
          {"approach": "aligned PatchCore + per-location z", "resolution": "896 (2x2 tiles)", "window": 2,
           "topk_frac": 0.001})
log_model("E1-896 raw (ablation: no location normalization)", route_no_history(s_E1_896_raw),
          {"approach": "aligned PatchCore, ratio to reference median", "resolution": "896 (2x2 tiles)"})

# COMMAND ----------

# DBTITLE 1,E2. New Dark Patch (pixel-level darkening vs. aligned dry references)
E2_DELTA = 8.0   # minimum L* drop (0-100 scale) to count as darker
E2_Z = 3.0       # and at least this many robust SDs of that pixel's day-to-day variation
E2_NOISE_FLOOR = 2.0


def e2_map(i, ref):
    """Signed darkening z-map of image i vs. its aligned references, robust to small shifts and lighting."""
    stack = np.stack([shift_to(LSTAR[j], *SHIFT[(i, j)]) for j in ref])
    valid = np.isfinite(stack).all(axis=0)
    med = np.nanmedian(stack, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(stack - med), axis=0)
    med = np.where(valid, med, 0); mad = np.where(valid, mad, 0)
    off = float(np.median((LSTAR[i] - med)[valid]))                 # global lighting change
    ref_dark = ndimage.minimum_filter(med, 5)                       # darkest nearby reference pixel (shift tolerance)
    noise = np.maximum(ndimage.maximum_filter(mad, 5), E2_NOISE_FLOOR)
    delta = (LSTAR[i] - off) - ref_dark
    z = np.where(valid, delta / noise, 0)
    mask = valid & (z < -E2_Z) & (delta < -E2_DELTA)
    return ndimage.binary_opening(mask, iterations=2), z


s_E2 = np.zeros(len(df)); E2_MASKS = {}
for i in range(len(df)):
    ref, fb = reference_idx(i)
    if fb:
        s_E2[i] = np.nan
        continue
    mask, _ = e2_map(i, ref)
    lab, n = ndimage.label(mask)
    s_E2[i] = float(ndimage.sum(mask, lab, range(1, n + 1)).max() / mask.size) if n else 0.0
    E2_MASKS[i] = mask
log_model("E2. New dark patch", route_no_history(s_E2),
          {"approach": "aligned pixel darkening blob", "delta_lstar": E2_DELTA, "z": E2_Z})

# COMMAND ----------

# DBTITLE 1,F, G. Ensembles
s_F = route_no_history((pct_rank(s_E1_896) + pct_rank(s_E2)) / 2)
s_G = route_no_history((pct_rank(s_B1) + pct_rank(s_C2) + pct_rank(s_E1_896) + pct_rank(s_E2)) / 4)
log_model("F. Ensemble E1-896 + E2", s_F, {"approach": "rank-average", "members": "E1-896,E2"})
log_model("G. Ensemble B1 + C2 + E1-896 + E2", s_G, {"approach": "rank-average", "members": "B1,C2,E1-896,E2"})

# COMMAND ----------

# DBTITLE 1,Per-Leak Ranks and Heatmaps
models = [k for k in RESULTS]
order = {k: (-SCORES[k].values).argsort(kind="stable").argsort() + 1 for k in models}
per_leak = SCORES.loc[y == 1, ["filename", "poi", "day", "no_history"]].copy()
for k in models:
    per_leak[k.split(".")[0]] = order[k][y == 1]
display(per_leak)

show = list(np.where((y == 1) & ~no_history)[0]) + [i for i in np.argsort(-s_F) if y[i] == 0 and not no_history[i]][:5]
fig, axes = plt.subplots(len(show), 3, figsize=(18, 3.4 * len(show)))
for row, i in zip(axes, show):
    img = Image.open(os.path.join(IMAGE_BASE_PATH, df.loc[i, "filename"])).convert("RGB").resize(SMALL)
    ref, _ = reference_idx(i)
    j = int(ref[np.argmax(CLS[ref] @ CLS[i])])
    row[0].imshow(Image.open(os.path.join(IMAGE_BASE_PATH, df.loc[j, "filename"])).convert("RGB").resize(SMALL))
    row[0].set_title(f"reference {day_arr[j]}", fontsize=9)
    zm = np.asarray(Image.fromarray(np.nan_to_num(Z896[i], nan=0)).resize(SMALL, Image.BILINEAR))
    row[1].imshow(img); row[1].imshow(np.clip(zm, 0, None), cmap="jet", alpha=0.45, vmin=0, vmax=12)
    row[1].set_title(f"{'LEAK' if y[i] else 'normal'} POI {poi_arr[i]} {day_arr[i]} | E1 rank {order['E1-896. Aligned PatchCore, location-normalized'][i]}",
                     fontsize=9)
    row[2].imshow(img); row[2].imshow(np.ma.masked_where(~E2_MASKS[i], E2_MASKS[i]), cmap="autumn", alpha=0.7)
    row[2].set_title(f"E2 dark blob | F rank {order['F. Ensemble E1-896 + E2'][i]}", fontsize=9)
    for ax in row:
        ax.axis("off")
plt.tight_layout()
plt.savefig("/tmp/v2_heatmaps.png", dpi=70)
plt.show()

# COMMAND ----------

# DBTITLE 1,Stage 1 Comparison Summary
cols = ["roc_auc", "pr_auc", "fp_to_catch_6_of_6", "fp_to_catch_5_of_6", "budget5_tp", "budget5_fp",
        "budget10_tp", "budget10_fp", "budget20_tp", "budget20_fp", "leak_ranks"]
summary = pd.DataFrame(RESULTS).T[cols]
display(summary)

with mlflow.start_run(run_name="Summary - Stage 1 v2 aligned experiments"):
    mlflow.log_table(summary.reset_index().rename(columns={"index": "model"}).astype(str), "summary.json")
    mlflow.log_table(per_leak.astype(str), "per_leak_ranks.json")
    mlflow.log_table(SCORES, "all_scores.json")
    mlflow.log_artifact("/tmp/v2_heatmaps.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2 — Vision-LLM verification
# MAGIC Candidates come from the `STAGE1_MODEL` score at the 10% and 20% false-alarm budgets. POIs without history are always
# MAGIC candidates. Each LLM sees the reference, current image, heatmap and up to two zoomed reference/current crop pairs.

# COMMAND ----------

# DBTITLE 1,Stage 2 Candidates (fixed from normal-image scores)
s1 = SCORES[STAGE1_MODEL].values
THR = {b: np.percentile(s1[y == 0], 100 - b) for b in (10, 20)}
cand_idx = np.where(s1 >= THR[20])[0]
for b, t in THR.items():
    sel = s1 >= t
    print(f"{b}% budget: {int(sel.sum())} candidates | {int((sel & (y == 1)).sum())} leaks | {int((sel & (y == 0)).sum())} normal")


def best_reference(i):
    ref, fb = reference_idx(i)
    return None if fb else int(ref[np.argmax(CLS[ref] @ CLS[i])])


CAND = pd.DataFrame({
    "idx": cand_idx, "filename": df.loc[cand_idx, "filename"].values, "poi": poi_arr[cand_idx],
    "day": day_arr[cand_idx], "leaked": y[cand_idx], "stage1_score": s1[cand_idx],
    "in_10pct": s1[cand_idx] >= THR[10], "reference_idx": [best_reference(i) for i in cand_idx],
})
CAND["has_reference"] = CAND["reference_idx"].notna()
display(CAND.drop(columns="idx"))

# COMMAND ----------

# DBTITLE 1,Build Images: Reference, Current, Heatmap and Zoomed Hotspot Crops
MAX_W = 1024
CROP = (1280, 720)   # crop size in full-resolution pixels (1/3 of a 4K frame per side)


def load_full(i):
    return Image.open(os.path.join(IMAGE_BASE_PATH, df.loc[i, "filename"])).convert("RGB")


def fit(img, max_w=MAX_W):
    return img.resize((max_w, int(img.height * max_w / img.width)), Image.BICUBIC) if img.width > max_w else img


def heatmap_overlay(img, i):
    z = Z896[i]
    hm = np.clip(np.nan_to_num(z, nan=0), 0, None)
    hm = hm / (hm.max() + 1e-9)
    hm = np.asarray(Image.fromarray(hm.astype(np.float32)).resize(img.size, Image.BILINEAR))
    color = (mpl_cm.jet(hm)[:, :, :3] * 255).astype(np.uint8)
    alpha = (0.15 + 0.5 * hm)[:, :, None]
    return Image.fromarray((np.asarray(img) * (1 - alpha) + color * alpha).astype(np.uint8))


def hotspots(i, W, H):
    """Up to two hotspot centers in full-resolution pixels: the E1 peak and the largest E2 dark blob."""
    pts = []
    z = Z896.get(i)
    if z is not None:
        zs = ndimage.uniform_filter(np.nan_to_num(z, nan=0), 2)
        r, c = np.unravel_index(np.argmax(zs), zs.shape)
        pts.append(((c + 0.5) * W / zs.shape[1], (r + 0.5) * H / zs.shape[0]))
    m = E2_MASKS.get(i)
    if m is not None and m.any():
        lab, n = ndimage.label(m)
        big = 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
        r, c = ndimage.center_of_mass(lab == big)
        p = (c * W / m.shape[1], r * H / m.shape[0])
        if not pts or np.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) > CROP[0] / 2:
            pts.append(p)
    return pts


def crop_at(img, cx, cy):
    x0 = int(np.clip(cx - CROP[0] / 2, 0, img.width - CROP[0])); y0 = int(np.clip(cy - CROP[1] / 2, 0, img.height - CROP[1]))
    return img.crop((x0, y0, x0 + CROP[0], y0 + CROP[1])), (x0, y0)


def b64(img):
    buf = io.BytesIO()
    fit(img).save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


IMAGES = {}
for _, r in CAND.iterrows():
    i = int(r["idx"])
    cur = load_full(i)
    W, H = cur.size
    ent = {"current": b64(cur), "heatmap": b64(heatmap_overlay(fit(cur), i)) if i in Z896 else None,
           "reference": None, "crops": []}
    if r["has_reference"]:
        j = int(r["reference_idx"])
        ref = load_full(j)
        ent["reference"] = b64(ref)
        dy, dx = SHIFT[(i, j)]   # cur[r, c] ≈ ref[r + dy, c + dx] at 960x540
        sx, sy = W / SMALL[0], H / SMALL[1]
        for cx, cy in hotspots(i, W, H):
            cc, (x0, y0) = crop_at(cur, cx, cy)
            rx0, ry0 = int(np.clip(x0 + dx * sx, 0, W - CROP[0])), int(np.clip(y0 + dy * sy, 0, H - CROP[1]))
            ent["crops"].append((b64(ref.crop((rx0, ry0, rx0 + CROP[0], ry0 + CROP[1]))), b64(cc)))
    IMAGES[i] = ent
print(f"Prepared images for {len(IMAGES)} candidates; crops per candidate: "
      f"{pd.Series([len(v['crops']) for v in IMAGES.values()]).value_counts().to_dict()}")

# COMMAND ----------

# DBTITLE 1,Verification Prompt
RULES = """Liquid leak indicators (must be NEW compared with the reference):
- A new dark or wet-looking patch on the floor, equipment, or fittings, often with a glossy/reflective sheen
- Pooling, droplets, drips, or a continuous flow path
- A smooth surface where floor texture was visible before; a sharp wet/dry boundary
- A wet area below a plausible source (flange, valve, joint, nozzle)
Small puddles count: a new wet patch only a few centimeters across is still a leak.

NOT a leak (common causes of false alerts):
- New or moved objects: buckets, drip trays, containers, tools, hoses, carts, cables, people
- Lighting, sunlight, reflections, or shadows that differ from the reference without a new wet patch
- Small differences in camera position, angle, or zoom
- Old stains, rust, painted markings, or residue that are also visible in the reference
- Uniform dirt, cleaning marks, footprints"""

OUTPUT = """Respond with ONLY a JSON object, no other text:
{"leak": true or false, "change_type": "liquid" | "object" | "lighting" | "camera_angle" | "old_stain" | "person" | "other" | "none", "confidence": number between 0 and 1, "evidence": "<max 200 characters>"}"""


def prompt_with_ref(n_crops):
    crops = "".join(
        f"\nImage {4 + 2 * k} — REFERENCE ZOOM {k + 1}: close-up of a hotspot in the reference (same area, camera-aligned).\n"
        f"Image {5 + 2 * k} — CURRENT ZOOM {k + 1}: the same close-up in the current image." for k in range(n_crops))
    return f"""You are verifying an automated leak alert from a robot inspection of an industrial plant.
The robot photographs the same point of interest repeatedly. A change detector compared the current image with earlier images of this exact spot and flagged a visual change.

Image 1 — REFERENCE: an earlier image of the same spot that a human confirmed has NO leak.
Image 2 — CURRENT: the image that triggered the alert.
Image 3 — CURRENT with change heatmap: red/yellow areas are where the detector saw the biggest difference from normal.{crops}

Compare the zoomed pairs carefully: they show the areas the detector found most changed, at full resolution.
Decide whether the change in the CURRENT image is a liquid leak or something else.
Call it a leak only if the current image shows a wet-looking area that is NOT present in the reference. If there is a genuinely new wet-looking area, call it a leak even if no source is visible.

{RULES}

{OUTPUT}"""


PROMPT_NO_REF = f"""You are verifying an automated leak alert from a robot inspection of an industrial plant.
This is a new inspection point: no earlier image of this spot exists, so judge the current image on its own.

Image 1 — CURRENT: the image to check.

Decide whether the current image shows a liquid leak.

{RULES}

{OUTPUT}"""


def build_messages(i):
    im = IMAGES[i]
    part = lambda b: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
    if im["reference"]:
        content = [part(im["reference"]), part(im["current"]), part(im["heatmap"])]
        for rc, cc in im["crops"]:
            content += [part(rc), part(cc)]
        content.append({"type": "text", "text": prompt_with_ref(len(im["crops"]))})
    else:
        content = [part(im["current"]), {"type": "text", "text": PROMPT_NO_REF}]
    return [{"role": "user", "content": content}]

# COMMAND ----------

# DBTITLE 1,LLM Client and Response Parsing
from openai import OpenAI

db_host = spark.conf.get("spark.databricks.workspaceUrl")
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
client = OpenAI(api_key=db_token, base_url=f"https://{db_host}/serving-endpoints")


def response_text(resp):
    content = resp.choices[0].message.content
    if isinstance(content, list):
        texts = [(b.text if hasattr(b, "text") else b.get("text", "")) for b in content
                 if (getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)) == "text"]
        return "\n".join(t for t in texts if t)
    return content or ""


def parse_verdict(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            return {"leak": bool(v.get("leak")), "change_type": str(v.get("change_type", "")),
                    "confidence": float(v.get("confidence", np.nan)), "evidence": str(v.get("evidence", ""))[:300],
                    "parse_ok": True}
        except Exception:
            pass
    leak = bool(re.search(r'"?leak"?\s*:\s*true', text, re.I))
    return {"leak": leak, "change_type": "", "confidence": np.nan, "evidence": text[:300], "parse_ok": False}


def verify(endpoint, i, retries=2):
    last_err = None
    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(model=endpoint, messages=build_messages(i), max_tokens=800)
            lat = (time.perf_counter() - t0) * 1000
            return {**parse_verdict(response_text(resp)), "latency_ms": lat, "error": None}
        except Exception as e:
            last_err = str(e)[:300]
            time.sleep(3 * (attempt + 1))
    # Fail open: an unverifiable alert is kept so a leak is never silently dropped
    return {"leak": True, "change_type": "error", "confidence": np.nan, "evidence": "",
            "parse_ok": False, "latency_ms": np.nan, "error": last_err}

# COMMAND ----------

# DBTITLE 1,Run Stage 2 for Each LLM and Log to MLflow
def two_stage_metrics(alerts_idx):
    pred = np.zeros(len(df), dtype=bool)
    pred[list(alerts_idx)] = True
    return at_threshold(pred.astype(float), y, 0.5)


VERDICTS = {}
STAGE2_RESULTS = []
for llm in LLMS:
    t0 = time.time()
    jobs = [(int(i), rep) for i in CAND["idx"] for rep in range(REPEATS)]
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda a: verify(llm["endpoint"], a[0]), jobs))
    raw = pd.DataFrame([{"idx": i, "repeat": rep, **o} for (i, rep), o in zip(jobs, outs)])
    agg = raw.groupby("idx").agg(leak_votes=("leak", "sum"), n=("leak", "size"),
                                 errors=("error", lambda s: int(s.notna().sum())),
                                 latency_ms=("latency_ms", "median")).reset_index()
    agg["leak"] = agg["leak_votes"] * 2 > agg["n"]
    agg["agreement"] = np.maximum(agg["leak_votes"], agg["n"] - agg["leak_votes"]) / agg["n"]
    # Explanation fields come from the first repeat that agrees with the majority
    rep = raw[raw["leak"] == raw["idx"].map(agg.set_index("idx")["leak"])].sort_values("repeat").drop_duplicates("idx")
    agg = agg.merge(rep[["idx", "change_type", "confidence", "evidence"]], on="idx", how="left")
    V = CAND.merge(agg, on="idx")
    VERDICTS[llm["name"]] = V
    print(f"\n{llm['name']}: {len(raw)} calls in {time.time() - t0:.0f}s, errors={int(raw['error'].notna().sum())}, "
          f"parse failures={int((~raw['parse_ok']).sum())}, unanimous={int((V['agreement'] == 1).sum())}/{len(V)}")

    for budget in (10, 20):
        sub = V[V["stage1_score"] >= THR[budget]]
        m = two_stage_metrics(sub.loc[sub["leak"], "idx"].astype(int))
        s1m = two_stage_metrics(sub["idx"].astype(int))
        # Per-repeat results, to show how much a single (non-voted) call varies
        per_rep = [two_stage_metrics(raw[(raw["repeat"] == r) & raw["leak"] & raw["idx"].isin(sub["idx"])]["idx"])
                   for r in range(REPEATS)]
        res = {"llm": llm["name"], "budget": budget, "stage1_tp": s1m["tp"], "stage1_fp": s1m["fp"],
               **{f"final_{k}": v for k, v in m.items()},
               "single_call_tp_range": f"{min(p['tp'] for p in per_rep)}-{max(p['tp'] for p in per_rep)}",
               "single_call_fp_range": f"{min(p['fp'] for p in per_rep)}-{max(p['fp'] for p in per_rep)}",
               "llm_calls": len(sub) * REPEATS,
               "p50_latency_ms": float(np.nanmedian(sub["latency_ms"])) if sub["latency_ms"].notna().any() else np.nan}
        STAGE2_RESULTS.append(res)
        with mlflow.start_run(run_name=f"Stage 2 v2: {STAGE1_MODEL.split('.')[0]} + {llm['name']} [{budget}% budget]"):
            mlflow.log_params({"stage1_model": STAGE1_MODEL, "stage1_budget_pct": budget, "stage2_llm": llm["name"],
                               "stage2_endpoint": llm["endpoint"], "repeats": REPEATS, "vote": "majority",
                               "images_sent": "reference + current + heatmap + aligned hotspot crop pairs",
                               "num_images": len(df), "num_leaks": int(y.sum())})
            mlflow.log_metrics({k: float(v) for k, v in res.items()
                                if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)})
            mlflow.log_table(sub.drop(columns=["idx", "reference_idx"]).astype(str), "candidate_verdicts.json")
            mlflow.log_dict({"with_reference_1_crop": prompt_with_ref(1), "no_reference": PROMPT_NO_REF}, "prompts.json")
        print(f"  {budget}% budget: stage 1 {s1m['tp']}/6 leaks, {s1m['fp']} FP -> final {m['tp']}/6 leaks, {m['fp']} FP "
              f"| F1 {m['f1']:.3f} | single call: TP {res['single_call_tp_range']}, FP {res['single_call_fp_range']}")

# COMMAND ----------

# DBTITLE 1,Final Comparison Summary (Stage 1 + Stage 2)
cur = at_threshold(df["current_alert"].values.astype(float), y, 0.5)
rows = [{"system": "Current production model", "leaks_caught": cur["tp"], "false_alarms": cur["fp"],
         "precision": cur["precision"], "recall": cur["recall"], "f1": cur["f1"], "llm_calls": 0}]
for b in (10, 20):
    m = two_stage_metrics(np.where(s1 >= THR[b])[0])
    rows.append({"system": f"Stage 1 only ({STAGE1_MODEL.split('.')[0]}) [{b}% budget]", "leaks_caught": m["tp"],
                 "false_alarms": m["fp"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "llm_calls": 0})
for r in STAGE2_RESULTS:
    rows.append({"system": f"{STAGE1_MODEL.split('.')[0]} + {r['llm']} [{r['budget']}% budget]",
                 "leaks_caught": r["final_tp"], "false_alarms": r["final_fp"], "precision": r["final_precision"],
                 "recall": r["final_recall"], "f1": r["final_f1"], "llm_calls": r["llm_calls"],
                 "single_call_tp": r["single_call_tp_range"], "single_call_fp": r["single_call_fp_range"]})
summary2 = pd.DataFrame(rows)
display(summary2)

detail = pd.concat([V.assign(llm=k) for k, V in VERDICTS.items()])[
    ["llm", "filename", "poi", "day", "leaked", "in_10pct", "has_reference", "leak", "leak_votes", "n", "agreement",
     "change_type", "confidence", "evidence"]]
display(detail.sort_values(["leaked", "llm"], ascending=[False, True]))

with mlflow.start_run(run_name=f"Summary - Stage 2 v2 LLM verification ({STAGE1_MODEL.split('.')[0]})"):
    mlflow.log_table(summary2.astype(str), "stage2_summary.json")
    mlflow.log_table(detail.astype(str), "stage2_all_verdicts.json")

dbutils.notebook.exit(json.dumps({
    "stage1_summary": summary.reset_index().rename(columns={"index": "model"}).astype(str).to_dict(orient="records"),
    "stage1_per_leak": per_leak.astype(str).to_dict(orient="records"),
    "stage2_summary": summary2.astype(str).to_dict(orient="records"),
    "leaks_and_kept_fp": detail[(detail["leaked"] == 1) | detail["leak"]].astype(str).to_dict(orient="records"),
    "shift_px": {"median": float(np.median(mags)), "p90": float(np.percentile(mags, 90)), "max": float(mags.max())},
}))
