"""Read DICOM headers (and a few pixel stats) for every series in the frozen download list.
    python src/extract_headers.py --stageII D:\\tcia\\stageII --ctc D:\\tcia\\ctc
Output: data/processed/dicom_series.parquet  (one row per series)
Needs:  pip install pydicom pandas numpy pyarrow  (pylibjpeg / gdcm only if some files are compressed)"""
import argparse, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, pydicom

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--stageII", required=True); ap.add_argument("--ctc", required=True)
ap.add_argument("--csv", default=str(ROOT / "src" / "frozen" / "download_series.csv"))
ap.add_argument("--split", default=str(ROOT / "data" / "processed" / "cohort_split.csv"))
ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "dicom_series.parquet"))
a = ap.parse_args()

dl = pd.read_csv(a.csv); split = pd.read_csv(a.split)[["patient_id", "split", "label"]]
dl = dl.drop(columns=["split"]).merge(split, on="patient_id", how="left")
TAGS = ["SeriesInstanceUID", "SliceThickness", "SpacingBetweenSlices", "PixelSpacing", "Rows", "Columns", "KVP",
        "XRayTubeCurrent", "ConvolutionKernel", "PatientPosition", "ContrastBolusAgent", "Manufacturer",
        "Modality", "BitsStored", "ImagePositionPatient", "InstanceNumber", "SeriesDescription",
        "ImageOrientationPatient", "RescaleSlope", "RescaleIntercept"]

def g(ds, k, default=None):
    v = getattr(ds, k, default)
    return default if v in ("", None) else v

def first(x):
    try: return float(x[0]) if hasattr(x, "__len__") and not isinstance(x, str) else float(x)
    except Exception: return np.nan

# 1) index: SeriesInstanceUID -> files
by_dir = defaultdict(list)
for root in (a.stageII, a.ctc):
    for f in Path(root).resolve().rglob("*.dcm"): by_dir[f.parent].append(f)
uid_dirs = {}
for d, files in by_dir.items():
    try: uid_dirs.setdefault(str(pydicom.dcmread(files[0], stop_before_pixels=True, specific_tags=["SeriesInstanceUID"]).SeriesInstanceUID), []).append(d)
    except Exception as e: print("unreadable", files[0], e)
print(f"{len(by_dir)} folders indexed")

rows = []
for i, r in enumerate(dl.itertuples(), 1):
    dirs = uid_dirs.get(r.series_uid)
    if not dirs: rows.append(dict(series_uid=r.series_uid, found=False)); continue
    files = [f for d in dirs for f in by_dir[d]]
    hdr = []
    for f in files:
        try: hdr.append(pydicom.dcmread(f, stop_before_pixels=True, specific_tags=TAGS))
        except Exception: pass
    pos = []
    for h in hdr:
        ipp = g(h, "ImagePositionPatient")
        if ipp is not None: pos.append((float(ipp[2]), f))
    z = np.sort(np.array([p[0] for p in pos])) if pos else np.array([])
    dz = np.diff(z) if len(z) > 1 else np.array([np.nan])
    h0 = hdr[0]
    row = dict(series_uid=r.series_uid, patient_id=r.patient_id, source=r.source, split=r.split, label=r.label,
               found=True, n_files=len(files), n_headers=len(hdr), folder=str(dirs[0]),
               desc=g(h0, "SeriesDescription"), manufacturer=g(h0, "Manufacturer"), modality=g(h0, "Modality"),
               rows=g(h0, "Rows"), cols=g(h0, "Columns"), pixel_mm=first(g(h0, "PixelSpacing", [np.nan])),
               slice_thickness=first(g(h0, "SliceThickness", np.nan)), spacing_between=first(g(h0, "SpacingBetweenSlices", np.nan)),
               z_step_median=float(np.nanmedian(dz)), z_step_max=float(np.nanmax(np.abs(dz))), z_extent_mm=float(z.max() - z.min()) if len(z) else np.nan,
               n_duplicate_z=int((np.abs(dz) < 1e-3).sum()) if len(z) > 1 else 0,
               kvp=first(g(h0, "KVP", np.nan)), tube_current=first(g(h0, "XRayTubeCurrent", np.nan)),
               kernel=str(g(h0, "ConvolutionKernel", "")), patient_position=g(h0, "PatientPosition"),
               contrast=g(h0, "ContrastBolusAgent"), bits=g(h0, "BitsStored"))
    # pixel stats from 3 middle slices (Hounsfield units)
    srt = [f for _, f in sorted(pos, key=lambda t: t[0])] or files
    pct, air, mid = [], [], None
    for f in [srt[len(srt) // 2 + k] for k in (-len(srt) // 6, 0, len(srt) // 6) if 0 <= len(srt) // 2 + k < len(srt)]:
        try:
            ds = pydicom.dcmread(f); arr = ds.pixel_array.astype(np.float32) * float(g(ds, "RescaleSlope", 1)) + float(g(ds, "RescaleIntercept", 0))
            pct.append(np.percentile(arr, [1, 50, 99])); air.append(float((arr < -900).mean())); mid = mid or str(f)
        except Exception as e: row["pixel_error"] = str(e)[:80]
    if pct:
        p = np.mean(pct, axis=0); row.update(hu_p1=p[0], hu_p50=p[1], hu_p99=p[2], air_frac=float(np.mean(air)), mid_file=mid)
    rows.append(row)
    if i % 25 == 0: print(f"{i}/{len(dl)} series")
out = pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True, exist_ok=True); out.to_parquet(a.out)
print(f"wrote {a.out}: {int(out.found.sum())} of {len(out)} series found")
