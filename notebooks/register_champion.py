# Databricks notebook source
# DBTITLE 1,Notebook Title
# MAGIC %md
# MAGIC # Leak Detection — Crown a Champion and Register It
# MAGIC
# MAGIC Run `leak_detection_v2` first. This notebook:
# MAGIC
# MAGIC 1. **Builds a leaderboard** of every stage 2 v2 configuration logged to the MLflow experiment (stage 1 model × LLM ×
# MAGIC    budget × repeats), combining repeated runs of the same configuration.
# MAGIC 2. **Crowns a champion** with the deck's rule: clear the recall bar in *every* run, then fewest false alarms (worst run,
# MAGIC    then average), then fewest LLM calls, then lowest latency.
# MAGIC 3. **Packages the champion** as an MLflow pyfunc that runs the whole decision tree on new photos: camera alignment →
# MAGIC    E1 + E2 → score F → alert threshold → vision-LLM check with a majority vote. The model carries its own reference
# MAGIC    bank (each POI's dry photos as DINOv2 patches, L* images, per-location normalizers and downsized JPEGs for the LLM).
# MAGIC 4. **Registers it in Unity Catalog** and moves the `Champion` alias only if the new version is at least as good as
# MAGIC    the current champion.
# MAGIC 5. **Smoke-tests** the registered model by loading it through the alias and scoring real photos.
# MAGIC
# MAGIC **Production reference set:** at inference time, a photo is compared with *all* dry photos of its POI known at
# MAGIC registration (in evaluation, same-day photos were held out). Re-run this notebook to refresh the bank as more photos are
# MAGIC labeled.

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install -q openai databricks-sdk mlflow scipy matplotlib pillow torch torchvision transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Settings
import os
import io
import re
import json
import time
import base64
import shutil
import tempfile
import numpy as np
import pandas as pd
import mlflow
from PIL import Image
from scipy import ndimage

dbutils.widgets.text("EXPERIMENT", "/Users/jeffrey.pao@databricks.com/Leak Detection - POI Baseline Experiments")
dbutils.widgets.text("MODEL_NAME", "stable_classic_wg38i9_catalog.westlake_gas_leakage_2.leak_detection_v2")
dbutils.widgets.text("MIN_RECALL", "1.0")
dbutils.widgets.dropdown("PROMOTE", "true", ["true", "false"])
EXPERIMENT = dbutils.widgets.get("EXPERIMENT")
MODEL_NAME = dbutils.widgets.get("MODEL_NAME")
MIN_RECALL = float(dbutils.widgets.get("MIN_RECALL"))
PROMOTE = dbutils.widgets.get("PROMOTE") == "true"

SPLITS_TABLE = "stable_classic_wg38i9_catalog.westlake_gas_leakage_2.silver_liquid_leak_results_splits"
IMAGE_BASE_PATH = "/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/OneDrive_1_10-13-2026/"
CACHE = "/Volumes/stable_classic_wg38i9_catalog/westlake_gas_leakage_2/documents/experiments_cache/stage1_v2/"
SUPPORTED_STAGE1 = "F. Ensemble E1-896 + E2"   # the stage 1 model this pyfunc implements
E1_COL, E2_COL = "E1-896. Aligned PatchCore, location-normalized", "E2. New dark patch"
MIN_REF = 5

mlflow.set_experiment(EXPERIMENT)
client = mlflow.MlflowClient()
uc = mlflow.MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------

# DBTITLE 1,Leaderboard: Every Stage 2 v2 Configuration in the Experiment
exp = client.get_experiment_by_name(EXPERIMENT)
runs = client.search_runs([exp.experiment_id], "attributes.run_name LIKE 'Stage 2 v2:%'", max_results=1000)
R = pd.DataFrame([{
    "run_id": r.info.run_id, "start_time": r.info.start_time,
    "stage1_model": r.data.params["stage1_model"], "llm": r.data.params["stage2_llm"],
    "endpoint": r.data.params["stage2_endpoint"], "budget": int(r.data.params["stage1_budget_pct"]),
    "repeats": int(r.data.params.get("repeats", 1)),
    "recall": r.data.metrics["final_recall"], "false_alarms": r.data.metrics["final_fp"],
    "precision": r.data.metrics["final_precision"], "f1": r.data.metrics["final_f1"],
    "llm_calls": r.data.metrics["llm_calls"], "latency_ms": r.data.metrics.get("p50_latency_ms", np.nan),
} for r in runs])
assert len(R), f"No 'Stage 2 v2:' runs in {EXPERIMENT}. Run leak_detection_v2 first."

KEY = ["stage1_model", "llm", "endpoint", "budget", "repeats"]
board = R.groupby(KEY).agg(
    runs=("run_id", "size"), min_recall=("recall", "min"), max_false_alarms=("false_alarms", "max"),
    mean_false_alarms=("false_alarms", "mean"), mean_precision=("precision", "mean"), mean_f1=("f1", "mean"),
    llm_calls=("llm_calls", "max"), latency_ms=("latency_ms", "median"),
).reset_index()
board = board.merge(R.sort_values("start_time").groupby(KEY)["run_id"].last().rename("latest_run_id").reset_index(), on=KEY)
board["eligible"] = board["min_recall"] >= MIN_RECALL - 1e-9
# The deck's rule: clear the recall bar, then fewest false alarms (worst run first), then cost, then latency
RANK_BY = ["eligible", "max_false_alarms", "mean_false_alarms", "llm_calls", "latency_ms"]
board = board.sort_values(RANK_BY, ascending=[False, True, True, True, True]).reset_index(drop=True)
board.insert(0, "rank", range(1, len(board) + 1))
display(board)

assert board["eligible"].any(), f"No configuration catches at least {MIN_RECALL:.0%} of leaks in every run"
champ = board.iloc[0]
assert champ["stage1_model"] == SUPPORTED_STAGE1, (
    f"Champion uses stage 1 model {champ['stage1_model']!r}; this notebook packages {SUPPORTED_STAGE1!r} only")
print(f"Champion: {champ['stage1_model']} + {champ['llm']} [{champ['budget']}% budget, {champ['repeats']} votes] | "
      f"recall {champ['min_recall']:.0%} in all {champ['runs']} run(s) | false alarms {int(champ['max_false_alarms'])} "
      f"(worst run) | {int(champ['llm_calls'])} LLM calls per 156 images")

# COMMAND ----------

# DBTITLE 1,Load Labels, Cached Embeddings and the Evaluation Score Distribution
df = spark.table(SPLITS_TABLE).toPandas()
volume_files = set(os.listdir(IMAGE_BASE_PATH))
df = df[df["filename"].isin(volume_files)].drop_duplicates("filename").sort_values("filename").reset_index(drop=True)
assert json.load(open(os.path.join(CACHE, "filenames.json"))) == df["filename"].tolist(), \
    "Embedding cache doesn't match the current images. Re-run leak_detection_v2 first."
df["day"] = pd.to_datetime(df["created_on_date_time"]).dt.strftime("%Y-%m-%d")
df["poi"] = df["point_of_interest_id"].astype(int)
df["y"] = (df["leaked"] == 1.0).astype(int)
LSTAR = np.load(os.path.join(CACHE, "lstar.npy"))
CLS = np.load(os.path.join(CACHE, "cls.npy"))
P896 = np.load(os.path.join(CACHE, "p896.npy"))

# Leave-one-day-out scores logged by the stage 1 run of the same leak_detection_v2 execution as the champion run
champ_start = R.loc[R["run_id"] == champ["latest_run_id"], "start_time"].iloc[0]
s1_runs = client.search_runs([exp.experiment_id], "attributes.run_name = 'Summary - Stage 1 v2 aligned experiments'",
                             order_by=["attributes.start_time DESC"], max_results=50)
s1_run = next(r for r in s1_runs if r.info.start_time <= champ_start)
S = mlflow.load_table("all_scores.json", run_ids=[s1_run.info.run_id]).set_index("filename").loc[df["filename"]]
hist = ~S["no_history"].astype(bool).values
HIST_E1, HIST_E2 = S[E1_COL].values[hist], S[E2_COL].values[hist]
F_eval = S[SUPPORTED_STAGE1].values
THRESHOLD = float(np.percentile(F_eval[df["y"].values == 0], 100 - int(champ["budget"])))


def prank(v, arr):
    """Percentile rank with ties averaged, matching pandas rank(pct=True) used in evaluation."""
    return (float((arr < v).sum()) + (float((arr == v).sum()) + 1) / 2) / len(arr)


# The pyfunc recomputes F as the mean percentile rank against these distributions; check it matches evaluation
F_check = np.array([(prank(a, HIST_E1) + prank(b, HIST_E2)) / 2 for a, b in zip(HIST_E1, HIST_E2)])
assert np.allclose(F_check, F_eval[hist], atol=1e-9), "Score reconstruction doesn't match the evaluation run"
print(f"Stage 1 scores from run {s1_run.info.run_id} | threshold F >= {THRESHOLD:.4f} ({champ['budget']}% budget)")

# COMMAND ----------

# DBTITLE 1,Build the Production Reference Bank (all dry photos of each POI)
import torch
from transformers import AutoModel

SMALL = (960, 540)
WIN = np.outer(np.hanning(SMALL[1]), np.hanning(SMALL[0])).astype(np.float32)


def phase_shift(a, b):
    """Integer (dy, dx) such that a[r, c] ≈ b[r + dy, c + dx]."""
    Fa = np.fft.rfft2((a - a.mean()) * WIN); Fb = np.fft.rfft2((b - b.mean()) * WIN)
    Rr = Fa * np.conj(Fb)
    r = np.fft.irfft2(Rr / (np.abs(Rr) + 1e-9), s=a.shape)
    py, px = np.unravel_index(np.argmax(r), r.shape)
    py = py - a.shape[0] if py > a.shape[0] // 2 else py
    px = px - a.shape[1] if px > a.shape[1] // 2 else px
    return -int(py), -int(px)


def aligned_map(Q, P, dy, dx, R=2):
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


ART = tempfile.mkdtemp()
os.makedirs(os.path.join(ART, "bank", "refs"), exist_ok=True)
sy, sx = P896.shape[1] / SMALL[1], P896.shape[2] / SMALL[0]
bank_index = {}
t0 = time.time()
for poi in sorted(df["poi"].unique()):
    idx = np.where((df["poi"].values == poi) & (df["y"].values == 0))[0]
    if len(idx) < MIN_REF:
        continue
    days = df["day"].values[idx]
    T = [torch.from_numpy(P896[j].astype(np.float32)) for j in idx]
    # Per-location normalizer: each reference vs. the other references from other days (as in evaluation)
    loo = []
    for a in range(len(idx)):
        others = [b for b in range(len(idx)) if b != a and days[b] != days[a]]
        if not others:
            continue
        maps = []
        for b in others:
            dy, dx = phase_shift(LSTAR[idx[a]], LSTAR[idx[b]])
            maps.append(aligned_map(T[a], T[b], round(dy * sy), round(dx * sx)))
        loo.append(np.min(maps, axis=0))
    loo = np.stack(loo); loo[~np.isfinite(loo)] = np.nan
    mu = np.nanmedian(loo, axis=0)
    sd = ndimage.gaussian_filter(np.nan_to_num(1.4826 * np.nanmedian(np.abs(loo - mu), axis=0),
                                               nan=np.nanmedian(loo)), 1.0)
    sd = np.maximum(sd, 0.5 * np.median(sd))
    files = df["filename"].values[idx].tolist()
    np.savez(os.path.join(ART, "bank", f"poi_{poi}.npz"), patches=P896[idx], lstar=LSTAR[idx].astype(np.float16),
             cls=CLS[idx], mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    for f in files:   # downsized copies, used only for the LLM's reference views
        im = Image.open(os.path.join(IMAGE_BASE_PATH, f)).convert("RGB")
        im.resize((1920, round(im.height * 1920 / im.width)), Image.BICUBIC).save(
            os.path.join(ART, "bank", "refs", f), format="JPEG", quality=90)
    bank_index[int(poi)] = {"files": files, "days": days.tolist()}
    print(f"  POI {poi}: {len(idx)} dry reference photos")
print(f"Reference bank for {len(bank_index)} POIs built in {time.time() - t0:.0f}s")

config = {
    "stage1_model": SUPPORTED_STAGE1, "budget_pct": int(champ["budget"]), "threshold": THRESHOLD,
    "llm_name": champ["llm"], "llm_endpoint": champ["endpoint"], "repeats": int(champ["repeats"]),
    "min_ref": MIN_REF, "window": 2, "topk_frac": 0.001, "small": list(SMALL), "grid": list(P896.shape[1:3]),
    "e2_delta": 8.0, "e2_z": 3.0, "e2_noise_floor": 2.0, "crop": [1280, 720], "max_w": 1024,
    "bank": bank_index,
}
json.dump(config, open(os.path.join(ART, "config.json"), "w"))
np.savez(os.path.join(ART, "historical.npz"), e1=HIST_E1, e2=HIST_E2)
AutoModel.from_pretrained("facebook/dinov2-small").save_pretrained(os.path.join(ART, "dino_model"))
size_mb = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(ART) for f in fs) / 1e6
print(f"Artifacts: {size_mb:.0f} MB")

# COMMAND ----------

# DBTITLE 1,Champion Model: the Full Decision Tree as an MLflow Pyfunc
class LeakDetectionV2(mlflow.pyfunc.PythonModel):
    """Leak detection v2: aligned change detection (F = E1 + E2) + vision-LLM verification.

    Input:  image_path (str), poi_id (int)
    Params: verify (bool, default True): run the stage 2 LLM check on stage 1 alerts
    Output: one row per image with stage 1 scores, the alert decision, and the LLM verdict"""

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

    MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])

    def load_context(self, context):
        import torch
        from transformers import AutoModel
        self.cfg = json.load(open(context.artifacts["config"]))
        self.bank_dir = context.artifacts["bank"]
        h = np.load(context.artifacts["historical"])
        self.hist_e1, self.hist_e2 = h["e1"], h["e2"]
        torch.set_grad_enabled(False)
        self.dino = AutoModel.from_pretrained(context.artifacts["dino_model"]).eval()
        self.small = tuple(self.cfg["small"])
        self.win = np.outer(np.hanning(self.small[1]), np.hanning(self.small[0])).astype(np.float32)
        self.banks, self._client = {}, None

    # ---------------- image helpers
    @staticmethod
    def _lab_l(rgb):
        r, g, b = [rgb[:, :, c].astype(np.float32) / 255 for c in range(3)]
        yy = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.where(yy > 0.008856, 116 * np.cbrt(yy) - 16, 903.3 * yy).astype(np.float32)

    def _embed(self, img, short, tiles):
        import torch
        w, h = img.size
        s = short / min(w, h)
        nw = round(w * s / 14 / tiles) * 14 * tiles; nh = round(h * s / 14 / tiles) * 14 * tiles
        arr = (np.asarray(img.resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255 - self.MEAN) / self.STD
        th, tw = nh // tiles, nw // tiles
        crops = [arr[a * th:(a + 1) * th, b * tw:(b + 1) * tw] for a in range(tiles) for b in range(tiles)]
        out = self.dino(pixel_values=torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2).copy()).float()).last_hidden_state
        pt = torch.nn.functional.normalize(out[:, 1:], dim=-1).numpy()
        gh, gw = th // 14, tw // 14
        grid = np.zeros((nh // 14, nw // 14, pt.shape[-1]), dtype=np.float32)
        for t, (a, b) in enumerate([(a, b) for a in range(tiles) for b in range(tiles)]):
            grid[a * gh:(a + 1) * gh, b * gw:(b + 1) * gw] = pt[t].reshape(gh, gw, -1)
        cls = torch.nn.functional.normalize(out[0, 0], dim=0).numpy()
        return cls, grid

    def _phase_shift(self, a, b):
        Fa = np.fft.rfft2((a - a.mean()) * self.win); Fb = np.fft.rfft2((b - b.mean()) * self.win)
        Rr = Fa * np.conj(Fb)
        r = np.fft.irfft2(Rr / (np.abs(Rr) + 1e-9), s=a.shape)
        py, px = np.unravel_index(np.argmax(r), r.shape)
        py = py - a.shape[0] if py > a.shape[0] // 2 else py
        px = px - a.shape[1] if px > a.shape[1] // 2 else px
        return -int(py), -int(px)

    @staticmethod
    def _shift_to(a, dy, dx):
        out = np.full(a.shape, np.nan, dtype=np.float32)
        H, W = a.shape[:2]
        r0, r1 = max(0, -dy), min(H, H - dy); c0, c1 = max(0, -dx), min(W, W - dx)
        if r1 > r0 and c1 > c0:
            out[r0:r1, c0:c1] = a[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
        return out

    @staticmethod
    def _aligned_map(Q, P, dy, dx, R):
        import torch
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

    def _bank(self, poi):
        key = str(int(poi))
        if key not in self.cfg["bank"]:
            return None
        if key not in self.banks:
            z = np.load(os.path.join(self.bank_dir, f"poi_{key}.npz"))
            self.banks[key] = {k: z[k] for k in z.files}
        return self.banks[key]

    @staticmethod
    def _prank(v, arr):
        """Percentile rank with ties averaged, matching pandas rank(pct=True) used in evaluation."""
        return (float((arr < v).sum()) + (float((arr == v).sum()) + 1) / 2) / len(arr)

    # ---------------- stage 1
    def _stage1(self, img, poi):
        import torch
        cfg = self.cfg
        L = self._lab_l(np.asarray(img.resize(self.small, Image.BILINEAR)))
        cls, _ = self._embed(img, 448, 1)
        _, P = self._embed(img, 896, 2)
        bank = self._bank(poi)
        if bank is None:
            return {"no_history": True, "cls": cls}
        if list(P.shape[:2]) != cfg["grid"]:
            raise ValueError(f"Patch grid {P.shape[:2]} doesn't match the reference bank {cfg['grid']}: "
                             "images must have the same resolution as the training photos")
        sy, sx = P.shape[0] / self.small[1], P.shape[1] / self.small[0]
        Q = torch.from_numpy(P)
        shifts, maps = [], []
        for j in range(len(bank["lstar"])):
            dy, dx = self._phase_shift(L, bank["lstar"][j].astype(np.float32))
            shifts.append((dy, dx))
            maps.append(self._aligned_map(Q, torch.from_numpy(bank["patches"][j].astype(np.float32)),
                                          round(dy * sy), round(dx * sx), cfg["window"]))
        q = np.min(maps, axis=0); q = np.where(np.isfinite(q), q, np.nan)
        z = (q - bank["mu"]) / bank["sd"]
        v = z[np.isfinite(z)]
        k = max(2, int(round(z.size * cfg["topk_frac"])))
        e1 = float(np.sort(v)[-k:].mean())
        # E2: new dark patch vs. the aligned references
        stack = np.stack([self._shift_to(bank["lstar"][j].astype(np.float32), *shifts[j]) for j in range(len(shifts))])
        valid = np.isfinite(stack).all(axis=0)
        med = np.where(valid, np.nanmedian(stack, axis=0), 0)
        mad = np.where(valid, 1.4826 * np.nanmedian(np.abs(stack - med), axis=0), 0)
        off = float(np.median((L - med)[valid]))
        delta = (L - off) - ndimage.minimum_filter(med, 5)
        zz = np.where(valid, delta / np.maximum(ndimage.maximum_filter(mad, 5), cfg["e2_noise_floor"]), 0)
        mask = ndimage.binary_opening(valid & (zz < -cfg["e2_z"]) & (delta < -cfg["e2_delta"]), iterations=2)
        lab, n = ndimage.label(mask)
        e2 = float(ndimage.sum(mask, lab, range(1, n + 1)).max() / mask.size) if n else 0.0
        f = (self._prank(e1, self.hist_e1) + self._prank(e2, self.hist_e2)) / 2
        ref = int(np.argmax(bank["cls"] @ cls))
        return {"no_history": False, "e1": e1, "e2": e2, "f": f, "zmap": z, "mask": mask, "shifts": shifts, "ref": ref}

    # ---------------- stage 2
    def _llm(self):
        if self._client is None:
            from databricks.sdk import WorkspaceClient
            self._client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        return self._client

    def _b64(self, img):
        mw = self.cfg["max_w"]
        if img.width > mw:
            img = img.resize((mw, int(img.height * mw / img.width)), Image.BICUBIC)
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()

    def _heatmap(self, img, z):
        from matplotlib import cm
        img = img.resize((self.cfg["max_w"], int(img.height * self.cfg["max_w"] / img.width)), Image.BICUBIC)
        hm = np.clip(np.nan_to_num(z, nan=0), 0, None); hm = hm / (hm.max() + 1e-9)
        hm = np.asarray(Image.fromarray(hm.astype(np.float32)).resize(img.size, Image.BILINEAR))
        color = (cm.jet(hm)[:, :, :3] * 255).astype(np.uint8); alpha = (0.15 + 0.5 * hm)[:, :, None]
        return Image.fromarray((np.asarray(img) * (1 - alpha) + color * alpha).astype(np.uint8))

    def _hotspots(self, s1, W, H):
        cw = self.cfg["crop"][0]
        zs = ndimage.uniform_filter(np.nan_to_num(s1["zmap"], nan=0), 2)
        r, c = np.unravel_index(np.argmax(zs), zs.shape)
        pts = [((c + 0.5) * W / zs.shape[1], (r + 0.5) * H / zs.shape[0])]
        m = s1["mask"]
        if m.any():
            lab, n = ndimage.label(m)
            big = 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
            r, c = ndimage.center_of_mass(lab == big)
            p = (c * W / m.shape[1], r * H / m.shape[0])
            if np.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) > cw / 2:
                pts.append(p)
        return pts

    def _messages(self, img, s1, poi):
        part = lambda b: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
        if s1["no_history"]:
            text = ("You are verifying an automated leak alert from a robot inspection of an industrial plant.\n"
                    "This is a new inspection point: no earlier image of this spot exists, so judge the current image on its own.\n\n"
                    "Image 1 — CURRENT: the image to check.\n\nDecide whether the current image shows a liquid leak.\n\n"
                    f"{self.RULES}\n\n{self.OUTPUT}")
            return [{"role": "user", "content": [part(self._b64(img)), {"type": "text", "text": text}]}], None
        key = str(int(poi))
        ref_file = self.cfg["bank"][key]["files"][s1["ref"]]
        ref = Image.open(os.path.join(self.bank_dir, "refs", ref_file)).convert("RGB")
        W, H = img.size
        cw, ch = self.cfg["crop"]
        rs = ref.width / W   # reference JPEGs are stored downsized
        dy, dx = s1["shifts"][s1["ref"]]
        content = [part(self._b64(ref)), part(self._b64(img)), part(self._b64(self._heatmap(img, s1["zmap"])))]
        crops = []
        for cx, cy in self._hotspots(s1, W, H):
            x0 = int(np.clip(cx - cw / 2, 0, W - cw)); y0 = int(np.clip(cy - ch / 2, 0, H - ch))
            rx0 = int(np.clip(x0 + dx * W / self.small[0], 0, W - cw)); ry0 = int(np.clip(y0 + dy * H / self.small[1], 0, H - ch))
            rc = ref.crop((int(rx0 * rs), int(ry0 * rs), int((rx0 + cw) * rs), int((ry0 + ch) * rs)))
            crops += [part(self._b64(rc)), part(self._b64(img.crop((x0, y0, x0 + cw, y0 + ch))))]
        n = len(crops) // 2
        zooms = "".join(
            f"\nImage {4 + 2 * k} — REFERENCE ZOOM {k + 1}: close-up of a hotspot in the reference (same area, camera-aligned).\n"
            f"Image {5 + 2 * k} — CURRENT ZOOM {k + 1}: the same close-up in the current image." for k in range(n))
        text = f"""You are verifying an automated leak alert from a robot inspection of an industrial plant.
The robot photographs the same point of interest repeatedly. A change detector compared the current image with earlier images of this exact spot and flagged a visual change.

Image 1 — REFERENCE: an earlier image of the same spot that a human confirmed has NO leak.
Image 2 — CURRENT: the image that triggered the alert.
Image 3 — CURRENT with change heatmap: red/yellow areas are where the detector saw the biggest difference from normal.{zooms}

Compare the zoomed pairs carefully: they show the areas the detector found most changed, at full resolution.
Decide whether the change in the CURRENT image is a liquid leak or something else.
Call it a leak only if the current image shows a wet-looking area that is NOT present in the reference. If there is a genuinely new wet-looking area, call it a leak even if no source is visible.

{self.RULES}

{self.OUTPUT}"""
        return [{"role": "user", "content": content + crops + [{"type": "text", "text": text}]}], ref_file

    def _verify_once(self, messages, retries=2):
        err = ""
        for attempt in range(retries + 1):
            try:
                resp = self._llm().chat.completions.create(model=self.cfg["llm_endpoint"], messages=messages, max_tokens=800)
                content = resp.choices[0].message.content
                if isinstance(content, list):
                    content = "\n".join(getattr(b, "text", None) or (b.get("text", "") if isinstance(b, dict) else "")
                                        for b in content)
                m = re.search(r"\{.*\}", content or "", re.S)
                v = json.loads(m.group(0))
                return {"leak": bool(v.get("leak")), "change_type": str(v.get("change_type", "")),
                        "confidence": float(v.get("confidence", np.nan)), "evidence": str(v.get("evidence", ""))[:300]}
            except Exception as e:
                err = str(e)[:200]
                time.sleep(3 * (attempt + 1))
        # Fail open: an unverifiable alert is kept so a leak is never silently dropped
        return {"leak": True, "change_type": "error", "confidence": np.nan, "evidence": err}

    def predict(self, context, model_input, params=None):
        from concurrent.futures import ThreadPoolExecutor
        verify = bool((params or {}).get("verify", True))
        rows = []
        for _, r in model_input.iterrows():
            img = Image.open(r["image_path"]).convert("RGB")
            s1 = self._stage1(img, r["poi_id"])
            alert = s1["no_history"] or s1["f"] >= self.cfg["threshold"]
            row = {"poi_id": int(r["poi_id"]), "no_history": s1["no_history"],
                   "e1_score": s1.get("e1", np.nan), "e2_score": s1.get("e2", np.nan),
                   "stage1_score": s1.get("f", np.nan), "stage1_alert": bool(alert), "verified": False,
                   "leak": bool(alert), "leak_votes": 0, "n_votes": 0, "change_type": "", "confidence": np.nan,
                   "evidence": "", "reference_file": ""}
            if alert and verify:
                msgs, ref_file = self._messages(img, s1, r["poi_id"])
                with ThreadPoolExecutor(self.cfg["repeats"]) as ex:
                    votes = list(ex.map(lambda _: self._verify_once(msgs), range(self.cfg["repeats"])))
                n_leak = sum(v["leak"] for v in votes)
                majority = n_leak * 2 > len(votes)
                pick = next(v for v in votes if v["leak"] == majority)
                row.update({"verified": True, "leak": majority, "leak_votes": n_leak, "n_votes": len(votes),
                            "change_type": pick["change_type"], "confidence": pick["confidence"],
                            "evidence": pick["evidence"], "reference_file": ref_file or ""})
            rows.append(row)
        return pd.DataFrame(rows)

# COMMAND ----------

# DBTITLE 1,Log the Model, Register It in Unity Catalog, and Crown the Champion
from mlflow.models import infer_signature

config_tag = f"{champ['stage1_model'].split('.')[0]} + {champ['llm']} [{champ['budget']}%]"
example = pd.DataFrame({"image_path": [os.path.join(IMAGE_BASE_PATH, df.loc[0, "filename"])], "poi_id": [int(df.loc[0, "poi"])]})
example_out = pd.DataFrame([{"poi_id": 1, "no_history": False, "e1_score": 1.0, "e2_score": 0.0, "stage1_score": 0.5,
                             "stage1_alert": False, "verified": False, "leak": False, "leak_votes": 0, "n_votes": 0,
                             "change_type": "", "confidence": 0.5, "evidence": "", "reference_file": ""}])
signature = infer_signature(example, example_out, params={"verify": True})
crown = {"min_recall": float(champ["min_recall"]), "max_false_alarms": float(champ["max_false_alarms"]),
         "mean_false_alarms": float(champ["mean_false_alarms"]), "mean_f1": float(champ["mean_f1"]),
         "llm_calls": float(champ["llm_calls"]), "latency_ms": float(champ["latency_ms"]), "eval_runs": float(champ["runs"])}

with mlflow.start_run(run_name=f"Champion registration - {config_tag}") as run:
    mlflow.log_params({"stage1_model": champ["stage1_model"], "stage2_llm": champ["llm"],
                       "stage2_endpoint": champ["endpoint"], "budget_pct": int(champ["budget"]),
                       "repeats": int(champ["repeats"]), "threshold": THRESHOLD, "min_recall_bar": MIN_RECALL,
                       "source_stage2_run": champ["latest_run_id"], "source_stage1_run": s1_run.info.run_id,
                       "reference_pois": len(bank_index)})
    mlflow.log_metrics(crown)
    mlflow.log_table(board.astype(str), "leaderboard.json")
    info = mlflow.pyfunc.log_model(
        name="model", python_model=LeakDetectionV2(),
        artifacts={"config": os.path.join(ART, "config.json"), "bank": os.path.join(ART, "bank"),
                   "historical": os.path.join(ART, "historical.npz"), "dino_model": os.path.join(ART, "dino_model")},
        signature=signature, input_example=example,
        pip_requirements=["torch", "torchvision", "transformers", "numpy", "scipy", "Pillow", "pandas", "matplotlib",
                          "openai", "databricks-sdk"])
print(f"Logged {info.model_uri} (run {run.info.run_id})")

mlflow.set_registry_uri("databricks-uc")
mv = mlflow.register_model(info.model_uri, MODEL_NAME)
for k, v in {**crown, "config": config_tag, "llm_endpoint": champ["endpoint"], "registration_run": run.info.run_id}.items():
    uc.set_model_version_tag(MODEL_NAME, mv.version, k, str(v))
uc.update_model_version(MODEL_NAME, mv.version, description=(
    f"{config_tag}, {champ['repeats']}-vote majority. Caught {champ['min_recall']:.0%} of leaks in all "
    f"{champ['runs']} evaluation run(s), worst-case {int(champ['max_false_alarms'])} false alarms on 156 labeled images "
    f"(leave-one-day-out). Input: image_path, poi_id. Param: verify."))


def crown_key(tags):
    """Same ordering as the leaderboard: higher recall, then fewer false alarms, then fewer LLM calls."""
    return (-float(tags["min_recall"]), float(tags["max_false_alarms"]), float(tags["mean_false_alarms"]),
            float(tags["llm_calls"]))


try:
    current = uc.get_model_version_by_alias(MODEL_NAME, "Champion")
except Exception:
    current = None
new_key = crown_key({k: str(v) for k, v in crown.items()})
if current is None:
    decision, reason = True, "no current Champion"
elif not all(k in current.tags for k in ("min_recall", "max_false_alarms", "mean_false_alarms", "llm_calls")):
    decision, reason = True, f"current Champion v{current.version} has no crown metrics to compare"
else:
    decision = new_key <= crown_key(current.tags)
    reason = f"v{mv.version} {'ties or beats' if decision else 'does not beat'} current Champion v{current.version}"
if PROMOTE and decision:
    uc.set_registered_model_alias(MODEL_NAME, "Champion", mv.version)
    print(f"Crowned {MODEL_NAME} v{mv.version} as Champion ({reason})")
else:
    print(f"Registered {MODEL_NAME} v{mv.version}; Champion alias unchanged "
          f"({'PROMOTE is false' if not PROMOTE else reason})")

# COMMAND ----------

# DBTITLE 1,Smoke Test: Load the Registered Model and Score Real Photos
uri = f"models:/{MODEL_NAME}@Champion" if (PROMOTE and decision) else f"models:/{MODEL_NAME}/{mv.version}"
model = mlflow.pyfunc.load_model(uri)

# Every leak, the false alarms the evaluation flagged, a no-history POI, and a few ordinary dry photos
y = df["y"].values
picks = list(np.where(y == 1)[0])
for poi, day in ((412, "2026-09-11"), (417, "2026-09-08")):
    picks.append(int(np.where((y == 0) & (df["poi"].values == poi) & (df["day"].values == day))[0][0]))
picks += [int(np.where((y == 0) & (df["poi"].values == 398))[0][0])]
picks += list(np.random.default_rng(0).choice([i for i in np.where(y == 0)[0] if i not in picks], 3, replace=False))
batch = pd.DataFrame({"image_path": [os.path.join(IMAGE_BASE_PATH, df.loc[i, "filename"]) for i in picks],
                      "poi_id": df.loc[picks, "poi"].astype(int).values})
t0 = time.time()
out = model.predict(batch, params={"verify": True})
out.insert(0, "leaked", y[picks]); out.insert(0, "day", df.loc[picks, "day"].values)
display(out)
print(f"Scored {len(batch)} photos in {time.time() - t0:.0f}s | leaks flagged by stage 1: "
      f"{int(out.loc[out.leaked == 1, 'stage1_alert'].sum())}/{int((out.leaked == 1).sum())} | "
      f"confirmed by the LLM: {int(out.loc[out.leaked == 1, 'leak'].sum())}/{int((out.leaked == 1).sum())}")

dbutils.notebook.exit(json.dumps({
    "champion": config_tag, "model": MODEL_NAME, "version": mv.version, "crowned": bool(PROMOTE and decision),
    "reason": reason, "leaderboard": board.astype(str).to_dict(orient="records"),
    "smoke_test": out.drop(columns=["evidence"]).astype(str).to_dict(orient="records"),
    "evidence": out["evidence"].tolist(), "artifacts_mb": round(size_mb),
}))
