# Databricks notebook source
# DBTITLE 1,Notebook Title
# MAGIC %md
# MAGIC # Leak Detection — Stage 2 v2: Vision-LLM Verification with Zoomed Hotspot Crops
# MAGIC
# MAGIC Uses the stage 1 v2 scores cached by `Leak Detection - Stage 1 v2 Aligned Experiments` (run that first).
# MAGIC
# MAGIC Changes from stage 2 v1:
# MAGIC - Candidates come from the stage 1 v2 model chosen in `STAGE1_MODEL` (10% and 20% false-alarm budgets, set from
# MAGIC   normal-image scores only). POIs without history are always candidates.
# MAGIC - Besides the full reference / current / heatmap images, the LLM gets **zoomed, camera-aligned crops** of the
# MAGIC   reference and current image around the strongest hotspots (E1 peak, E2 dark blob). Small puddles are only a few
# MAGIC   dozen pixels in a downscaled 4K frame.
# MAGIC - Each LLM is called `REPEATS` times per candidate; the verdict is the majority vote, and the agreement rate is
# MAGIC   reported so run-to-run variability is visible.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install -q openai mlflow scikit-learn pillow matplotlib
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
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from scipy import ndimage
from matplotlib import cm as mpl_cm

EXPERIMENT = "/Users/jeffrey.pao@databricks.com/Leak Detection - POI Baseline Experiments"
mlflow.set_experiment(EXPERIMENT)

SPLITS_TABLE = "stable_classic_wg38i9_catalog.westlake_gas_leakage_2.silver_liquid_leak_results_splits"
IMAGE_BASE_PATH = "/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/OneDrive_1_10-13-2026/"
CACHE = "/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/experiments_cache/stage1_v2/"
dbutils.widgets.text("STAGE1_MODEL", "F. Ensemble E1-896 + E2")
dbutils.widgets.text("LLMS", "Gemini 3.8 Flash=databricks-gemini-3-8-flash,Claude Sonnet 5.5=databricks-claude-sonnet-5-5")
dbutils.widgets.text("REPEATS", "3")
STAGE1_MODEL = dbutils.widgets.get("STAGE1_MODEL")
LLMS = [dict(zip(("name", "endpoint"), x.split("="))) for x in dbutils.widgets.get("LLMS").split(",")]
REPEATS = int(dbutils.widgets.get("REPEATS"))
MIN_REF = 5
SMALL = (960, 540)

df = spark.table(SPLITS_TABLE).toPandas()
volume_files = set(os.listdir(IMAGE_BASE_PATH))
df = df[df["filename"].isin(volume_files)].drop_duplicates("filename").sort_values("filename").reset_index(drop=True)
assert json.load(open(os.path.join(CACHE, "filenames.json"))) == df["filename"].tolist(), "stage 1 v2 cache is stale"
df["day"] = pd.to_datetime(df["created_on_date_time"]).dt.strftime("%Y-%m-%d")
df["poi"] = df["point_of_interest_id"].astype(int)
df["y"] = (df["leaked"] == 1.0).astype(int)
df["current_alert"] = (df["leak_suspected"] == "Yes").astype(int)
y = df["y"].values
poi_arr = df["poi"].values
day_arr = df["day"].values
normal_arr = y == 0

SCORES = pd.read_csv(os.path.join(CACHE, "scores.csv"))
CLS = np.load(os.path.join(CACHE, "cls.npy"))
ZMAPS = dict(np.load(os.path.join(CACHE, "e1_896_zmaps.npz")))
E2_MASKS = dict(np.load(os.path.join(CACHE, "e2_masks.npz")))
SHIFT = {tuple(map(int, k.split(","))): tuple(v) for k, v in json.load(open(os.path.join(CACHE, "shifts.json")))["shift"].items()}
s1 = SCORES[STAGE1_MODEL].values
print(f"{len(df)} images | {int(y.sum())} leaks | stage 1 model: {STAGE1_MODEL} | LLMs: {[l['name'] for l in LLMS]}")


def reference_idx(i):
    base = normal_arr & (day_arr != day_arr[i])
    same = base & (poi_arr == poi_arr[i])
    if same.sum() >= MIN_REF:
        return np.where(same)[0], False
    return np.where(base)[0], True


def at_threshold(scores, y, thr):
    pred = scores >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, alerts=tp + fp, precision=prec, recall=rec, f1=f1)

# COMMAND ----------

# DBTITLE 1,Stage 2 Candidates (fixed from normal-image scores)
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
    z = ZMAPS[str(i)]
    hm = np.clip(np.nan_to_num(z, nan=0), 0, None)
    hm = hm / (hm.max() + 1e-9)
    hm = np.asarray(Image.fromarray(hm.astype(np.float32)).resize(img.size, Image.BILINEAR))
    color = (mpl_cm.jet(hm)[:, :, :3] * 255).astype(np.uint8)
    alpha = (0.15 + 0.5 * hm)[:, :, None]
    return Image.fromarray((np.asarray(img) * (1 - alpha) + color * alpha).astype(np.uint8))


def hotspots(i, W, H):
    """Up to two hotspot centers in full-resolution pixels: the E1 peak and the largest E2 dark blob."""
    pts = []
    z = ZMAPS.get(str(i))
    if z is not None:
        zs = ndimage.uniform_filter(np.nan_to_num(z, nan=0), 2)
        r, c = np.unravel_index(np.argmax(zs), zs.shape)
        pts.append(((c + 0.5) * W / zs.shape[1], (r + 0.5) * H / zs.shape[0]))
    m = E2_MASKS.get(str(i))
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
    ent = {"current": b64(cur), "heatmap": b64(heatmap_overlay(fit(cur), i)) if str(i) in ZMAPS else None,
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

# DBTITLE 1,Comparison Summary
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
    "summary": summary2.astype(str).to_dict(orient="records"),
    "leaks_and_kept_fp": detail[(detail["leaked"] == 1) | detail["leak"]].astype(str).to_dict(orient="records"),
}))
