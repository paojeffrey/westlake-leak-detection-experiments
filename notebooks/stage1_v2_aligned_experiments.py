# Databricks notebook source
# DBTITLE 1,Notebook Title
# MAGIC %md
# MAGIC # Leak Detection — Stage 1 v2: Aligned, Location-Normalized Change Detection
# MAGIC
# MAGIC Model D (B1 + PatchCore-lite) misses the POI 422 leak on 2026-09-14 (rank 94 of 156). That leak is a small
# MAGIC dark puddle (~340×50 px in a 4K frame), which covers only ~5 of ~1,800 DINOv2 patches at 448 px. PatchCore-lite
# MAGIC scores the mean of the top 1% (~18) patches against an *unaligned* bank, so the puddle is diluted by ordinary
# MAGIC background patches (windows, lights, reflections) that vary from day to day.
# MAGIC
# MAGIC New candidates (all unsupervised, same leave-one-day-out protocol):
# MAGIC
# MAGIC | Model | Idea |
# MAGIC |---|---|
# MAGIC | E1-448 / E1-896 | **Aligned PatchCore**: estimate the camera shift to each dry reference, compare each patch only with the same location (±small window), then z-score each location against how much it varies across the POI's dry images. Score = mean of top 0.1% z. 896 = 2×2 tiles at 448 (2× resolution). |
# MAGIC | E1-896 raw | Ablation: aligned distances without per-location normalization |
# MAGIC | E2 | **New dark patch**: pixel-level L* darkening vs. the aligned dry references, robust to small shifts and lighting. Score = area of largest darkened blob. |
# MAGIC | F | Rank-average of E1-896 + E2 |
# MAGIC | G | Rank-average of B1 + C2 + E1-896 + E2 |
# MAGIC
# MAGIC **POIs without history** (fewer than 5 dry images of the same POI) cannot be change-detected. For E/F/G they are
# MAGIC routed to stage 2 unconditionally ("new POI → always verify") and placed at the top of the ranking.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install -q mlflow scikit-learn scipy matplotlib pillow torch torchvision transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup and Data Loading
import os
import time
import json
import numpy as np
import pandas as pd
import mlflow
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score, average_precision_score

EXPERIMENT = "/Users/jeffrey.pao@databricks.com/Leak Detection - POI Baseline Experiments"
mlflow.set_experiment(EXPERIMENT)

SPLITS_TABLE = "stable_classic_wg38i9_catalog.westlake_gas_leakage_2.silver_liquid_leak_results_splits"
IMAGE_BASE_PATH = "/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/OneDrive_1_10-13-2026/"
# Cached embeddings, maps and scores (shared with the stage 2 v2 notebook)
CACHE = "/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/experiments_cache/stage1_v2/"
os.makedirs(CACHE, exist_ok=True)
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
plt.savefig(os.path.join(CACHE, "v2_heatmaps.png"), dpi=70)
plt.show()

# COMMAND ----------

# DBTITLE 1,Comparison Summary and Cache for Stage 2
cols = ["roc_auc", "pr_auc", "fp_to_catch_6_of_6", "fp_to_catch_5_of_6", "budget5_tp", "budget5_fp",
        "budget10_tp", "budget10_fp", "budget20_tp", "budget20_fp", "leak_ranks"]
summary = pd.DataFrame(RESULTS).T[cols]
display(summary)

SCORES.to_csv(os.path.join(CACHE, "scores.csv"), index=False)
# Volumes don't support the seeks that zip writes need: write locally, then copy
import shutil, tempfile
for name, arrs in (("e1_896_zmaps.npz", Z896), ("e2_masks.npz", E2_MASKS)):
    tmp = os.path.join(tempfile.mkdtemp(), name)
    np.savez_compressed(tmp, **{str(i): a for i, a in arrs.items()})
    shutil.copy(tmp, os.path.join(CACHE, name))
json.dump({"shift": {f"{i},{j}": v for (i, j), v in SHIFT.items()}}, open(os.path.join(CACHE, "shifts.json"), "w"))

with mlflow.start_run(run_name="Summary - Stage 1 v2 aligned experiments"):
    mlflow.log_table(summary.reset_index().rename(columns={"index": "model"}).astype(str), "summary.json")
    mlflow.log_table(per_leak.astype(str), "per_leak_ranks.json")
    mlflow.log_artifact(os.path.join(CACHE, "v2_heatmaps.png"))

dbutils.notebook.exit(json.dumps({
    "summary": summary.reset_index().rename(columns={"index": "model"}).astype(str).to_dict(orient="records"),
    "per_leak": per_leak.astype(str).to_dict(orient="records"),
    "shift_px": {"median": float(np.median(mags)), "p90": float(np.percentile(mags, 90)), "max": float(mags.max())},
}))
