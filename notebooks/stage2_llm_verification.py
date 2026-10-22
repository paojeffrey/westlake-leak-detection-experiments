# Databricks notebook source
# DBTITLE 1,Notebook Title
# MAGIC %md
# MAGIC # Leak Detection — Stage 2: Vision-LLM Verification of Change Alerts
# MAGIC
# MAGIC **Stage 1** (unchanged from the POI baseline experiment): ensemble D = POI-relative features (B1) + DINOv2
# MAGIC PatchCore-lite (C2). It flags images that look different from the same POI's normal history.
# MAGIC
# MAGIC **Stage 2** (this notebook): each flagged image is sent to a vision LLM together with
# MAGIC 1. the most similar **known-dry reference image of the same POI** from another day, and
# MAGIC 2. the **change heatmap** from stage 1,
# MAGIC
# MAGIC and the LLM decides whether the change is **liquid** or something else (new object, lighting, angle, old stain…).
# MAGIC
# MAGIC Candidate sets are fixed from normal-image scores only (no tuning on leaks):
# MAGIC - **10% budget**: stage-1 threshold where 10% of normal images alert (prior result: 4/6 leaks, 15 false alarms)
# MAGIC - **20% budget**: wider net so stage 2 has a chance to keep 5/6 leaks
# MAGIC
# MAGIC Stage 2 can only remove alerts, so its recall is capped by stage 1.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install -q openai mlflow scikit-learn scipy matplotlib pillow torch torchvision transformers
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

EXPERIMENT = "/Users/jeffrey.pao@databricks.com/Leak Detection - POI Baseline Experiments"
mlflow.set_experiment(EXPERIMENT)

SPLITS_TABLE = "stable_classic_wg38i9_catalog.westlake_gas_leakage_2.silver_liquid_leak_results_splits"
IMAGE_BASE_PATH = "/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/OneDrive_1_10-13-2026/"
MIN_REF = 5
RNG = np.random.default_rng(42)

df = spark.table(SPLITS_TABLE).toPandas()
volume_files = set(os.listdir(IMAGE_BASE_PATH))
df = df[df["filename"].isin(volume_files)].drop_duplicates("filename").reset_index(drop=True)
df["ts"] = pd.to_datetime(df["created_on_date_time"])
df["day"] = df["ts"].dt.strftime("%Y-%m-%d")
df["poi"] = df["point_of_interest_id"].astype(int)
df["y"] = (df["leaked"] == 1.0).astype(int)
df["current_alert"] = (df["leak_suspected"] == "Yes").astype(int)

y = df["y"].values
N_LEAKS = int(y.sum())
N_NORMAL = int((y == 0).sum())
print(f"{len(df)} images | {N_LEAKS} leaks | {N_NORMAL} normal | {df['poi'].nunique()} POIs | {df['day'].nunique()} days")
display(df[df.y == 1][["filename", "poi", "ts", "split", "leak_suspected"]])

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


fallback = np.array([reference_idx(i)[1] for i in range(len(df))])
print(f"Images using all-POI fallback reference: {fallback.sum()} "
      f"(leaks: {int(y[fallback].sum())}, POIs: {sorted(set(poi_arr[fallback]))})")


def fp_at_recall(scores, y, k):
    """False alarms you must accept to catch the top-k leaks."""
    leak_scores = np.sort(scores[y == 1])[::-1]
    if k > len(leak_scores) or not np.isfinite(leak_scores[k - 1]):
        return np.nan
    thr = leak_scores[k - 1]
    return int(((scores >= thr) & (y == 0)).sum())


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
    m = {
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
        "fp_to_catch_6_of_6": fp_at_recall(scores, y, 6),
        "fp_to_catch_5_of_6": fp_at_recall(scores, y, 5),
        "fp_to_catch_4_of_6": fp_at_recall(scores, y, 4),
        "fp_to_catch_3_of_6": fp_at_recall(scores, y, 3),
    }
    # Fixed false-alarm budgets, set from normal-image scores only (no leak labels used)
    for budget in (5, 10):
        thr = np.percentile(scores[y == 0], 100 - budget)
        for k, v in at_threshold(scores, y, thr).items():
            m[f"budget{budget}_{k}"] = v
    # Rank of each leak among all 157 images (1 = most suspicious)
    order = (-scores).argsort().argsort() + 1
    m["leak_ranks"] = sorted(int(r) for r in order[y == 1])
    return m


RESULTS = {}
SCORES = pd.DataFrame({"filename": df["filename"], "poi": df["poi"], "day": df["day"], "leaked": y})


def log_model(name, scores, params, artifacts=None):
    m = evaluate(scores, y)
    RESULTS[name] = m
    SCORES[name] = scores
    with mlflow.start_run(run_name=name):
        mlflow.log_params({**params, "num_images": len(df), "num_leaks": N_LEAKS,
                           "eval_protocol": "leave-one-day-out", "min_poi_reference": MIN_REF})
        mlflow.log_metrics({k: float(v) for k, v in m.items()
                            if k != "leak_ranks" and v is not None and np.isfinite(v)})
        mlflow.log_dict({"leak_ranks": m["leak_ranks"]}, "leak_ranks.json")
        per_img = SCORES[["filename", "poi", "day", "leaked", name]].rename(columns={name: "score"})
        mlflow.log_table(per_img.sort_values("score", ascending=False), "per_image_scores.json")
        for path in artifacts or []:
            mlflow.log_artifact(path)
    print(f"{name}: ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} | "
          f"FP to catch 6/6={m['fp_to_catch_6_of_6']} 5/6={m['fp_to_catch_5_of_6']} 4/6={m['fp_to_catch_4_of_6']} | "
          f"5% budget: {m['budget5_tp']}/6 caught, {m['budget5_fp']} FP | leak ranks={m['leak_ranks']}")
    return m

# COMMAND ----------

# DBTITLE 1,Feature Extraction (same 10 features as the hybrid notebook)
def rgb_to_sv(rgb):
    r, g, b = [rgb[:, :, c].astype(float) for c in range(3)]
    maxc = np.maximum(np.maximum(r, g), b); minc = np.minimum(np.minimum(r, g), b)
    return np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-9), 0), maxc


def lab_l(rgb):
    r, g, b = [rgb[:, :, c].astype(float) / 255 for c in range(3)]
    yy = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.where(yy > 0.008856, 116 * np.cbrt(yy) - 16, 903.3 * yy)


def extract_features(path):
    img = Image.open(path).convert("RGB")
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
    return {
        "reflectivity_mean": float(v_ch.mean()),
        "reflectivity_std": float(v_ch.std()),
        "dark_patch_area": float(dark.mean()),
        "significant_dark_components": int(sum(1 for s in sizes if s > dark.size * 0.002)),
        "texture_variance": float(np.var(lap)),
        "local_var_mean": float(np.mean(lsq - lm ** 2)),
        "edge_density": float(emask.mean()),
        "boundary_edge_density": float((emask & bmask).sum() / max(bmask.sum(), 1)),
        "color_anomaly": float(np.mean(np.abs(L - np.median(L)))),
        "dark_saturation_mean": float(s_ch[dark > 0].mean()) if dark.sum() else 0.0,
    }


t0 = time.time()
feat = pd.DataFrame([extract_features(os.path.join(IMAGE_BASE_PATH, f)) for f in df["filename"]])
FEATURES = list(feat.columns)
F = feat.values.astype(float)
print(f"Extracted {F.shape[1]} features from {F.shape[0]} images in {time.time() - t0:.1f}s")

# COMMAND ----------

# DBTITLE 1,POI-Relative Feature Transform
def poi_relative(F, exclude_days=()):
    """Robust z-score of every feature vs. the image's POI reference (median / MAD)."""
    Z = np.zeros_like(F)
    for i in range(len(F)):
        ref, _ = reference_idx(i, exclude_days)
        R = F[ref]
        med = np.median(R, axis=0)
        mad = 1.4826 * np.median(np.abs(R - med), axis=0)
        scale = np.where(mad > 1e-9, mad, R.std(axis=0) + 1e-9)
        Z[i] = (F[i] - med) / scale
    return np.clip(Z, -10, 10)


Z_all = poi_relative(F)
print("POI-relative features computed.")

# COMMAND ----------

# DBTITLE 1,DINOv2 Embeddings (CLS + patch tokens)
import torch
from transformers import AutoModel

torch.set_grad_enabled(False)
dino = AutoModel.from_pretrained("facebook/dinov2-small").eval()
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])


def prep(img, short=448):
    w, h = img.size
    s = short / min(w, h)
    nw, nh = max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)
    arr = (np.asarray(img.resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0), (nh // 14, nw // 14)


CLS, PATCHES, GRIDS = [], [], []
t0 = time.time()
for f in df["filename"]:
    x, grid = prep(Image.open(os.path.join(IMAGE_BASE_PATH, f)).convert("RGB"))
    out = dino(pixel_values=x).last_hidden_state[0]
    cls = torch.nn.functional.normalize(out[0], dim=0)
    pt = torch.nn.functional.normalize(out[1:], dim=1)
    CLS.append(cls.numpy()); PATCHES.append(pt.half().numpy()); GRIDS.append(grid)
CLS = np.stack(CLS)
print(f"Embedded {len(CLS)} images in {time.time() - t0:.1f}s; patch grid e.g. {GRIDS[0]}")

# COMMAND ----------

# DBTITLE 1,Stage 1: B1 + PatchCore-lite + Ensemble D (identical to the baseline experiment run)
s_B1 = np.sqrt(np.mean(Z_all ** 2, axis=1))


def normalize_by_reference(raw):
    out = np.zeros_like(raw)
    for i in range(len(raw)):
        ref, _ = reference_idx(i)
        out[i] = raw[i] / (np.median(raw[ref]) + 1e-9)
    return out


MAX_BANK = 30000
TOPK_FRAC = 0.01
PATCH_MAPS = {}


def patch_nn_dist(i, ref):
    bank = np.concatenate([PATCHES[j] for j in ref]).astype(np.float32)
    if len(bank) > MAX_BANK:
        bank = bank[RNG.choice(len(bank), MAX_BANK, replace=False)]
    q = torch.from_numpy(PATCHES[i].astype(np.float32))
    b = torch.from_numpy(bank)
    d = torch.full((len(q),), float("inf"))
    for s in range(0, len(b), 10000):
        d = torch.minimum(d, torch.cdist(q, b[s:s + 10000]).min(dim=1).values)
    return d.numpy()


raw_C2 = np.zeros(len(df))
t0 = time.time()
for i in range(len(df)):
    ref, _ = reference_idx(i)
    d = patch_nn_dist(i, ref)
    k = max(1, int(len(d) * TOPK_FRAC))
    raw_C2[i] = float(np.sort(d)[-k:].mean())
    PATCH_MAPS[i] = d.reshape(GRIDS[i])
print(f"PatchCore-lite scored {len(df)} images in {time.time() - t0:.1f}s")

s_C2 = normalize_by_reference(raw_C2)


def pct_rank(s):
    return pd.Series(s).rank(pct=True).values


s_D = (pct_rank(s_B1) + pct_rank(s_C2)) / 2
m_D = evaluate(s_D, y)
print(f"Stage 1 (D) check — leak ranks {m_D['leak_ranks']}, 10% budget: {m_D['budget10_tp']}/6 caught, {m_D['budget10_fp']} FP")

# COMMAND ----------

# DBTITLE 1,Stage 2 Candidates (fixed from normal-image scores)
THR = {b: np.percentile(s_D[y == 0], 100 - b) for b in (10, 20)}
cand_idx = np.where(s_D >= THR[20])[0]
for b, t in THR.items():
    sel = s_D >= t
    print(f"{b}% budget: {int(sel.sum())} candidates | {int((sel & (y == 1)).sum())} leaks | {int((sel & (y == 0)).sum())} normal")


def best_reference(i):
    """Most similar known-dry image of the same POI from another day (None if the POI has no history)."""
    ref, fb = reference_idx(i)
    if fb:
        return None
    return int(ref[np.argmax(CLS[ref] @ CLS[i])])


CAND = pd.DataFrame({
    "idx": cand_idx,
    "filename": df.loc[cand_idx, "filename"].values,
    "poi": poi_arr[cand_idx],
    "day": day_arr[cand_idx],
    "leaked": y[cand_idx],
    "stage1_score": s_D[cand_idx],
    "in_10pct": s_D[cand_idx] >= THR[10],
    "reference_idx": [best_reference(i) for i in cand_idx],
})
CAND["has_reference"] = CAND["reference_idx"].notna()
display(CAND.drop(columns="idx"))

# COMMAND ----------

# DBTITLE 1,Build Reference / Current / Heatmap Images
import io
import base64
from matplotlib import cm as mpl_cm

MAX_W = 1024


def load_rgb(i):
    img = Image.open(os.path.join(IMAGE_BASE_PATH, df.loc[i, "filename"])).convert("RGB")
    if img.width > MAX_W:
        img = img.resize((MAX_W, int(img.height * MAX_W / img.width)), Image.BICUBIC)
    return img


def heatmap_overlay(i):
    img = load_rgb(i)
    hm = PATCH_MAPS[i].astype(np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    hm = np.asarray(Image.fromarray(hm).resize(img.size, Image.BILINEAR))
    color = (mpl_cm.jet(hm)[:, :, :3] * 255).astype(np.uint8)
    alpha = (0.15 + 0.5 * hm)[:, :, None]
    return Image.fromarray((np.asarray(img) * (1 - alpha) + color * alpha).astype(np.uint8))


def b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


IMAGES = {}
for _, r in CAND.iterrows():
    i = int(r["idx"])
    IMAGES[i] = {
        "reference": b64(load_rgb(int(r["reference_idx"]))) if r["has_reference"] else None,
        "current": b64(load_rgb(i)),
        "heatmap": b64(heatmap_overlay(i)),
    }
print(f"Prepared images for {len(IMAGES)} candidates")

# Example triplet for the notebook reader
ex = CAND.sort_values("leaked", ascending=False).iloc[0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (title, im) in zip(axes, [("Reference (dry)", load_rgb(int(ex["reference_idx"])) if ex["has_reference"] else load_rgb(int(ex["idx"]))),
                                  ("Current", load_rgb(int(ex["idx"]))), ("Change heatmap", heatmap_overlay(int(ex["idx"])))]):
    ax.imshow(im); ax.set_title(title); ax.axis("off")
plt.suptitle(f"What the LLM sees — POI {ex['poi']} {ex['day']} ({'LEAK' if ex['leaked'] else 'normal'})")
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,Verification Prompt
RULES = """Liquid leak indicators (must be NEW compared with the reference):
- A new dark or wet-looking patch on the floor, equipment, or fittings, often with a glossy/reflective sheen
- Pooling, droplets, drips, or a continuous flow path
- A smooth surface where floor texture was visible before; a sharp wet/dry boundary
- A wet area below a plausible source (flange, valve, joint, nozzle)

NOT a leak (common causes of false alerts):
- New or moved objects: buckets, drip trays, containers, tools, hoses, carts, cables, people
- Lighting, sunlight, reflections, or shadows that differ from the reference without a new wet patch
- Small differences in camera position, angle, or zoom
- Old stains, rust, or residue that are also visible in the reference
- Uniform dirt, cleaning marks, footprints"""

OUTPUT = """Respond with ONLY a JSON object, no other text:
{"leak": true or false, "change_type": "liquid" | "object" | "lighting" | "camera_angle" | "old_stain" | "person" | "other" | "none", "confidence": number between 0 and 1, "evidence": "<max 200 characters>"}"""

PROMPT_WITH_REF = f"""You are verifying an automated leak alert from a robot inspection of an industrial plant.
The robot photographs the same point of interest repeatedly. A change detector compared the current image with earlier images of this exact spot and flagged a visual change.

Image 1 — REFERENCE: an earlier image of the same spot that a human confirmed has NO leak.
Image 2 — CURRENT: the image that triggered the alert.
Image 3 — CURRENT with change heatmap: red/yellow areas are where the detector saw the biggest difference from normal.

Decide whether the change in the CURRENT image is a liquid leak or something else.
Call it a leak only if the current image shows a wet-looking area that is NOT present in the reference. If there is a genuinely new wet-looking area, call it a leak even if no source is visible.

{RULES}

{OUTPUT}"""

PROMPT_NO_REF = f"""You are verifying an automated leak alert from a robot inspection of an industrial plant.
No earlier reference image exists for this spot, so judge the current image on its own.

Image 1 — CURRENT: the image that triggered the alert.
Image 2 — CURRENT with change heatmap: red/yellow areas are where the detector saw the biggest difference from typical plant images.

Decide whether the current image shows a liquid leak.

{RULES}

{OUTPUT}"""


def build_messages(i):
    im = IMAGES[i]
    def part(b):
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
    if im["reference"]:
        content = [part(im["reference"]), part(im["current"]), part(im["heatmap"]), {"type": "text", "text": PROMPT_WITH_REF}]
    else:
        content = [part(im["current"]), part(im["heatmap"]), {"type": "text", "text": PROMPT_NO_REF}]
    return [{"role": "user", "content": content}]

# COMMAND ----------

# DBTITLE 1,LLM Client and Response Parsing
import re
from openai import OpenAI

db_host = spark.conf.get("spark.databricks.workspaceUrl")
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
client = OpenAI(api_key=db_token, base_url=f"https://{db_host}/serving-endpoints")

LLMS = [
    {"name": "Claude Sonnet 5.5", "endpoint": "databricks-claude-sonnet-5-5"},
    {"name": "Claude Opus 4.7", "endpoint": "databricks-claude-opus-4-7"},
    {"name": "Gemini 3.8 Flash", "endpoint": "databricks-gemini-3-8-flash"},
    {"name": "GPT-6.1 Sol", "endpoint": "databricks-gpt-6-1-sol"},
]


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
            resp = client.chat.completions.create(model=endpoint, messages=build_messages(i), max_tokens=600)
            lat = (time.perf_counter() - t0) * 1000
            return {**parse_verdict(response_text(resp)), "latency_ms": lat, "error": None}
        except Exception as e:
            last_err = str(e)[:300]
            time.sleep(3 * (attempt + 1))
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
    rows = []
    for _, r in CAND.iterrows():
        v = verify(llm["endpoint"], int(r["idx"]))
        rows.append({**r.to_dict(), **v})
    V = pd.DataFrame(rows)
    VERDICTS[llm["name"]] = V
    n_err = int(V["error"].notna().sum())
    print(f"\n{llm['name']}: {len(V)} calls in {time.time() - t0:.0f}s, errors={n_err}, parse failures={int((~V['parse_ok']).sum())}")

    for budget in (10, 20):
        sub = V[V["stage1_score"] >= THR[budget]]
        kept = sub[sub["leak"]]
        m = two_stage_metrics(kept["idx"].astype(int))
        s1 = two_stage_metrics(sub["idx"].astype(int))
        res = {"llm": llm["name"], "budget": budget,
               "stage1_tp": s1["tp"], "stage1_fp": s1["fp"],
               **{f"final_{k}": v for k, v in m.items()},
               "fp_removed": s1["fp"] - m["fp"], "leaks_lost": s1["tp"] - m["tp"],
               "llm_calls": len(sub), "llm_errors": int(sub["error"].notna().sum()),
               "p50_latency_ms": float(np.nanmedian(sub["latency_ms"])) if sub["latency_ms"].notna().any() else np.nan}
        STAGE2_RESULTS.append(res)
        with mlflow.start_run(run_name=f"Stage 2: D + {llm['name']} verifier [{budget}% budget]"):
            mlflow.log_params({"stage1_model": "D. Ensemble B1 + C2", "stage1_budget_pct": budget,
                               "stage2_llm": llm["name"], "stage2_endpoint": llm["endpoint"],
                               "reference_selection": "most similar normal image, same POI, other day (DINOv2 CLS)",
                               "images_sent": "reference + current + heatmap (current + heatmap if no POI history)",
                               "num_images": len(df), "num_leaks": N_LEAKS, "eval_protocol": "leave-one-day-out"})
            mlflow.log_metrics({k: float(v) for k, v in res.items()
                                if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)})
            mlflow.log_table(sub.drop(columns=["idx", "reference_idx"]).assign(
                reference_file=[df.loc[int(j), "filename"] if pd.notna(j) else None for j in sub["reference_idx"]]),
                "candidate_verdicts.json")
            mlflow.log_dict({"with_reference": PROMPT_WITH_REF, "no_reference": PROMPT_NO_REF}, "prompts.json")
        print(f"  {budget}% budget: stage 1 {s1['tp']}/6 leaks, {s1['fp']} FP -> final {m['tp']}/6 leaks, "
              f"{m['fp']} FP | precision {m['precision']:.1%} recall {m['recall']:.1%} F1 {m['f1']:.3f}")

# COMMAND ----------

# DBTITLE 1,Comparison Summary
cur = at_threshold(df["current_alert"].values.astype(float), y, 0.5)
base = [{"system": "Current production model", "leaks_caught": cur["tp"], "false_alarms": cur["fp"],
         "precision": cur["precision"], "recall": cur["recall"], "f1": cur["f1"], "llm_calls": len(df)}]
for b in (10, 20):
    s1 = two_stage_metrics(np.where(s_D >= THR[b])[0])
    base.append({"system": f"Stage 1 only (D) [{b}% budget]", "leaks_caught": s1["tp"], "false_alarms": s1["fp"],
                 "precision": s1["precision"], "recall": s1["recall"], "f1": s1["f1"], "llm_calls": 0})
for r in STAGE2_RESULTS:
    base.append({"system": f"D + {r['llm']} [{r['budget']}% budget]", "leaks_caught": r["final_tp"],
                 "false_alarms": r["final_fp"], "precision": r["final_precision"], "recall": r["final_recall"],
                 "f1": r["final_f1"], "llm_calls": r["llm_calls"]})
summary2 = pd.DataFrame(base)
display(summary2)

# Why were false alarms kept / leaks dropped?
detail = pd.concat([V.assign(llm=k) for k, V in VERDICTS.items()])
detail = detail[["llm", "filename", "poi", "day", "leaked", "in_10pct", "has_reference", "leak", "change_type",
                 "confidence", "evidence", "error"]]
display(detail.sort_values(["leaked", "llm"], ascending=[False, True]))

with mlflow.start_run(run_name="Summary - Stage 2 LLM verification"):
    mlflow.log_table(summary2, "stage2_summary.json")
    mlflow.log_table(detail, "stage2_all_verdicts.json")

dbutils.notebook.exit(json.dumps({
    "summary": summary2.astype(str).to_dict(orient="records"),
    "detail": detail.astype(str).to_dict(orient="records"),
}))
