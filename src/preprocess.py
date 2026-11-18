"""Harmonise every selected volume and save intensity features for the source-prediction check.
    python src/preprocess.py --out D:\\tcia\\processed          (add --limit 6 to try a few series first)
Per series:  load -> sort slices along the patient axis -> common orientation (anatomical frame)
             -> average into 5 mm slabs -> resample in-plane to 1.5 mm -> body mask (padding, table, outside air removed)
             -> clip HU to [-1024, 1000] -> crop to body -> save float16 .npy
Output: <out>/<series_uid>.npy and data/processed/preproc_series.parquet (features raw vs preprocessed).
Resumable: series whose .npy already exists are skipped (their features are recomputed from the file).
Needs: pip install pydicom numpy pandas scipy pyarrow"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, pydicom
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--series", default=str(ROOT / "data/processed/dicom_series.parquet"))
ap.add_argument("--result", default=str(ROOT / "data/processed/preproc_series.parquet"))
ap.add_argument("--min-slices", type=int, default=200)    # CTC volume rule (section 1b of the image notebook)
ap.add_argument("--max-step", type=float, default=2.5)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

SLAB_MM, PIX_MM, HU_LO, HU_HI, SIZE = 5.0, 1.5, -1024.0, 1000.0, 256
RAW_EDGES = np.linspace(-3100, 1500, 33)           # 32 bins, shows padding values
PRE_EDGES = np.linspace(HU_LO, 400, 33)            # 32 bins in the clipped range

def load_volume(files):
    """Return volume (z, y, x) in HU in an anatomical frame, z positions (mm, feet -> head), pixel size (mm)."""
    first = pydicom.dcmread(files[0], stop_before_pixels=True)
    iop = np.array(getattr(first, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0]), float); r, c = iop[:3], iop[3:]; n = np.cross(r, c)
    items = []
    for f in files:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        items.append((float(np.dot(n, np.array(ds.ImagePositionPatient, float))), arr))
    items.sort(key=lambda t: t[0])
    z = np.array([t[0] for t in items]); keep = np.r_[True, np.diff(z) > 1e-3]       # drop duplicate positions
    vol = np.stack([t[1] for t, k in zip(items, keep) if k]); z = z[keep]
    if r[0] < 0: vol = vol[:, :, ::-1]            # columns increase toward patient left
    if c[1] < 0: vol = vol[:, ::-1, :]            # rows increase toward patient posterior
    return vol, z, float(first.PixelSpacing[0])

def slabs(vol, z):
    """Average slices into 5 mm bins (empty bins dropped)."""
    b = np.round((z - z[0]) / SLAB_MM).astype(int); u, inv = np.unique(b, return_inverse=True)
    out = np.zeros((len(u),) + vol.shape[1:], np.float32); cnt = np.bincount(inv).astype(np.float32)
    for k in range(len(vol)): out[inv[k]] += vol[k]
    return out / cnt[:, None, None]

def fit(vol):
    """Centre-crop / pad (with -1024) to SIZE x SIZE."""
    h, w = vol.shape[1:]; out = np.full((vol.shape[0], SIZE, SIZE), HU_LO, np.float32)
    y0, x0 = max((h - SIZE) // 2, 0), max((w - SIZE) // 2, 0); oy, ox = max((SIZE - h) // 2, 0), max((SIZE - w) // 2, 0)
    ch, cw = min(h, SIZE), min(w, SIZE); out[:, oy:oy + ch, ox:ox + cw] = vol[:, y0:y0 + ch, x0:x0 + cw]
    return out

def body_mask(vol):
    m = np.zeros(vol.shape, bool)
    for i, s in enumerate(vol):
        b = ndi.binary_opening(s > -300, iterations=3)           # detach the table
        lab, n = ndi.label(b)
        if n: b = lab == (1 + np.argmax(ndi.sum(b, lab, range(1, n + 1))))
        m[i] = ndi.binary_fill_holes(b)                            # colon air stays inside
    return m

def features(body_vals):
    p = np.percentile(body_vals, [1, 50, 99]); h, _ = np.histogram(np.clip(body_vals, PRE_EDGES[0], PRE_EDGES[-1]), bins=PRE_EDGES)
    d = dict(pre_p1=p[0], pre_p50=p[1], pre_p99=p[2], pre_air_frac=float((body_vals < -900).mean()))
    d.update({f"pre_h{i:02d}": v for i, v in enumerate(h / max(h.sum(), 1))}); return d

d = pd.read_parquet(a.series); d = d[d.found == True]
keep = (d.source == "StageII") | ((d.n_files >= a.min_slices) & (d.z_step_median <= a.max_step))
d = d[keep]
if a.limit: d = pd.concat([d[d.source == "StageII"].head(a.limit // 2), d[d.source == "CTC"].head(a.limit - a.limit // 2)])
out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
rows = []
for i, r in enumerate(d.itertuples(), 1):
    npy = out_dir / f"{r.series_uid}.npy"
    try:
        files = sorted(Path(r.folder).glob("*.dcm"))
        vol, z, px = load_volume(files)
        raw_h, _ = np.histogram(vol, bins=RAW_EDGES); raw = {f"raw_h{k:02d}": v / raw_h.sum() for k, v in enumerate(raw_h)}
        raw.update(raw_min=float(vol.min()), raw_pad_frac=float((vol <= -1500).mean()))
        sl = slabs(vol, z); del vol
        sl = ndi.zoom(sl, (1, px / PIX_MM, px / PIX_MM), order=1); sl = fit(sl)
        m = body_mask(sl); sl = np.where(m, np.clip(sl, HU_LO, HU_HI), HU_LO)
        ys, xs = np.where(m.any(0))
        if len(ys) == 0: raise ValueError("no body found")
        crop = sl[:, ys.min():ys.max() + 1, xs.min():xs.max() + 1]; mc = m[:, ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        np.save(npy, crop.astype(np.float16))
        row = dict(series_uid=r.series_uid, ok=True, shape=str(crop.shape), body_frac=float(mc.mean()), out_file=str(npy))
        row.update(raw); row.update(features(crop[mc]))
    except Exception as e:
        row = dict(series_uid=r.series_uid, ok=False, error=str(e)[:200]); print("FAILED", r.series_uid[-10:], e)
    rows.append(row)
    if i % 10 == 0 or i == len(d): print(f"{i}/{len(d)}", flush=True)
res = pd.DataFrame(rows); Path(a.result).parent.mkdir(parents=True, exist_ok=True); res.to_parquet(a.result)
print(f"wrote {a.result}: {int(res.ok.sum())} ok, {int((~res.ok).sum())} failed")
