"""MSD Task10 Colon: the same-hospital positive control.  Can the pipeline tell tumour-containing colon slabs from
tumour-free colon slabs when every scan comes from ONE hospital (so the source cannot give the answer away)?

Everything large goes on the D: drive (--root, default D:\\msd).  Only small CSVs go in the project folder.

    1) download  (about 5-6 GB; resumable: run it again if the connection drops)
       python src/msd_colon.py download
    2) prepare   (NIfTI -> the same 5 mm slab / 1.5 mm pixel / body-cropped float16 .npy as preprocess.py; ~20-40 min)
       python src/msd_colon.py prepare
    3) colon     (TotalSegmentator colon masks, same settings as select_colon.py, then slab labels and the split)
       python src/msd_colon.py colon --device gpu --fast

Space needed on D:  ~6 GB while downloading, ~12 GB during extraction (the .tar is deleted afterwards unless
--keep-tar), ~2 GB for the prepared scans.  The download step checks free space first.

Data: Medical Segmentation Decathlon, Task10_Colon (Simpson et al., 2019; Antonelli et al., 2022), CC-BY-SA 4.0,
126 labelled CT scans of primary colon cancer from Memorial Sloan Kettering (one hospital), tumour masks per voxel.
Only imagesTr/labelsTr are used (imagesTs has no labels).  Hidden '._' files from the archive are skipped.

Slab labels (step 3), per 5 mm slab inside the colon range:
    tumor     >= --min-tumor-px tumour pixels (default 20 px = 45 mm^2 at 1.5 mm)
    no_tumor  0 tumour pixels AND at least --gap slabs (default 2 = 10 mm) away from any tumour slab
    dropped   everything in between (a few tumour pixels, or right next to the tumour: partial-volume edge)
Patient-level split 70 / 15 / 15 with the project seed 20261023 (every patient has a tumour, so it is not stratified).

Outputs
    D:\\msd\\processed\\<case>.npy               prepared scan (slab, row, col), float16 HU, same frame as preprocess.py
    D:\\msd\\processed\\tumor\\<case>.npz        tumour mask on the prepared grid (bool)
    D:\\msd\\processed\\colon\\<case>.npz        colon mask on the prepared grid (bool)
    data/processed/msd_preproc.csv             one row per case: ok, shape, tumour slabs, file
    data/processed/msd_colon_slices.csv        one row per kept slab: patient_id, source, label, split, series_uid,
                                               slab_idx, colon_area_px, tumor_px, n_slabs   (same columns as colon_slices.csv
                                               plus tumor_px; 'series_uid' = case id, so existing loaders work)
    data/processed/msd_colon_series.csv        one row per case: colon range, slab counts per label, status
Needs: pip install nibabel numpy pandas scipy   (+ totalsegmentator for step 3, already installed for select_colon.py)
"""
import argparse, http.client, shutil, sys, tarfile, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar"
SLAB_MM, PIX_MM, HU_LO, HU_HI, SIZE = 5.0, 1.5, -1024.0, 1000.0, 256      # must match preprocess.py
SEED = 20261023


# ---------------------------------------------------------------- 1) download
def free_gb(path):
    p = Path(path)
    while not p.exists(): p = p.parent
    return shutil.disk_usage(p).free / 1e9


def download(a):
    root = Path(a.root)
    if not Path(root.anchor).exists(): raise SystemExit(f"cannot reach {root.anchor} (is the D: drive connected?)")
    root.mkdir(parents=True, exist_ok=True)
    tar = root / "Task10_Colon.tar"; done = root / "Task10_Colon" / "dataset.json"
    if done.exists(): print(f"already extracted: {done.parent}"); return
    have = tar.stat().st_size if tar.exists() else 0
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r: total = int(r.headers.get("Content-Length", 0))
    need = (total - have) / 1e9 + total / 1e9 + 2            # rest of download + extraction + prepared scans
    print(f"archive {total / 1e9:.2f} GB, already have {have / 1e9:.2f} GB, free on {root.anchor} {free_gb(root):.1f} GB")
    if free_gb(root) < need: raise SystemExit(f"need about {need:.0f} GB free on {root.anchor}; free some space and run again")
    tries = 0
    while have < total:                          # a dropped or stalled connection resumes from where it stopped
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r, open(tar, "ab" if have else "wb") as f:
                if have and r.status != 206: raise SystemExit("server ignored the resume request; delete the .tar and run again")
                t0, start, got = time.time(), have, have
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk: break
                    f.write(chunk); got += len(chunk)
                    if got // (200 << 20) != (got - len(chunk)) // (200 << 20):
                        mbps = (got - start) / 1e6 / max(time.time() - t0, 1)
                        print(f"  {got / 1e9:.2f} / {total / 1e9:.2f} GB  ({mbps:.1f} MB/s)", flush=True)
        except (OSError, http.client.HTTPException) as e:          # timeouts, resets, incomplete reads
            tries += 1
            if tries > 30: raise SystemExit(f"download keeps failing ({e}); run the same command again later to resume")
            print(f"  connection problem ({type(e).__name__}); resuming in 10 s (retry {tries}/30)", flush=True); time.sleep(10)
        have = tar.stat().st_size if tar.exists() else 0
    if tar.stat().st_size != total: raise SystemExit("download incomplete; run the same command again to resume")
    print("extracting imagesTr, labelsTr, dataset.json ...")
    base = root.resolve()
    with tarfile.open(tar) as t:
        keep = []
        for m in t.getmembers():
            name = Path(m.name).name
            if name.startswith("._") or not (m.isfile() or m.isdir()): continue
            parts = Path(m.name).parts
            if not (len(parts) >= 2 and parts[1] in ("imagesTr", "labelsTr", "dataset.json")): continue
            if not (base / m.name).resolve().is_relative_to(base): raise SystemExit(f"unsafe path in archive: {m.name}")
            keep.append(m)
        if hasattr(tarfile, "data_filter"): t.extractall(base, members=keep, filter="data")
        else: t.extractall(base, members=keep)
    if not done.exists(): raise SystemExit("extraction did not produce Task10_Colon/dataset.json")
    if not a.keep_tar: tar.unlink(); print("deleted the .tar to save space")
    n = len(list((root / "Task10_Colon" / "imagesTr").glob("colon_*.nii.gz")))
    print(f"done: {n} labelled scans in {root / 'Task10_Colon'}")


# ---------------------------------------------------------------- 2) prepare
def to_pipeline_frame(img):
    """NIfTI -> (slab, row, col) with col -> patient left, row -> posterior, slab -> head (the preprocess.py frame)."""
    import nibabel as nib
    c = nib.as_closest_canonical(img)                         # RAS+: x -> right, y -> anterior, z -> superior
    d = np.asanyarray(c.dataobj)
    zooms = c.header.get_zooms()[:3]
    arr = d[::-1, ::-1, :].transpose(2, 1, 0)                 # flip x (-> left), flip y (-> posterior), (z, y, x)
    return np.ascontiguousarray(arr), (float(zooms[2]), float(zooms[1]), float(zooms[0]))


def slab_bins(nz, dz):
    b = np.round(np.arange(nz) * dz / SLAB_MM).astype(int); return np.unique(b, return_inverse=True)[1]


def to_slabs(vol, inv, reduce):
    n = inv.max() + 1; out = np.zeros((n,) + vol.shape[1:], np.float32)
    if reduce == "mean":
        cnt = np.bincount(inv).astype(np.float32)
        for k in range(len(vol)): out[inv[k]] += vol[k]
        return out / cnt[:, None, None]
    for k in range(len(vol)): out[inv[k]] = np.maximum(out[inv[k]], vol[k])
    return out


def fit(vol, fill):
    h, w = vol.shape[1:]; out = np.full((vol.shape[0], SIZE, SIZE), fill, np.float32)
    y0, x0 = max((h - SIZE) // 2, 0), max((w - SIZE) // 2, 0); oy, ox = max((SIZE - h) // 2, 0), max((SIZE - w) // 2, 0)
    ch, cw = min(h, SIZE), min(w, SIZE); out[:, oy:oy + ch, ox:ox + cw] = vol[:, y0:y0 + ch, x0:x0 + cw]
    return out


def body_mask(vol):
    m = np.zeros(vol.shape, bool)
    for i, s in enumerate(vol):
        b = ndi.binary_opening(s > -300, iterations=3); lab, n = ndi.label(b)
        if n: b = lab == (1 + np.argmax(ndi.sum(b, lab, range(1, n + 1))))
        m[i] = ndi.binary_fill_holes(b)
    return m


def prepare_case(img_path, lab_path):
    import nibabel as nib
    vol, (dz, dy, dx) = to_pipeline_frame(nib.load(str(img_path))); vol = vol.astype(np.float32)
    lab, _ = to_pipeline_frame(nib.load(str(lab_path))); lab = (lab > 0).astype(np.float32)
    if lab.shape != vol.shape: raise ValueError(f"label shape {lab.shape} != image shape {vol.shape}")
    inv = slab_bins(vol.shape[0], dz)
    sl, tl = to_slabs(vol, inv, "mean"), to_slabs(lab, inv, "max")
    sl = ndi.zoom(sl, (1, dy / PIX_MM, dx / PIX_MM), order=1); tl = ndi.zoom(tl, (1, dy / PIX_MM, dx / PIX_MM), order=0)
    sl, tl = fit(sl, HU_LO), fit(tl, 0) > 0.5
    m = body_mask(sl); sl = np.where(m, np.clip(sl, HU_LO, HU_HI), HU_LO)
    ys, xs = np.where(m.any(0))
    if len(ys) == 0: raise ValueError("no body found")
    box = (slice(None), slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return sl[box].astype(np.float16), tl[box] & m[box]


def prepare(a):
    src = Path(a.root) / "Task10_Colon"; out = Path(a.root) / "processed"; (out / "tumor").mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (src / "imagesTr").glob("colon_*.nii.gz") if not p.name.startswith("._"))
    if not imgs: raise SystemExit(f"no scans in {src / 'imagesTr'}; run the download step first")
    if a.limit: imgs = imgs[:a.limit]
    rows = []
    for i, p in enumerate(imgs, 1):
        case = p.name.replace(".nii.gz", ""); npy, tz = out / f"{case}.npy", out / "tumor" / f"{case}.npz"
        try:
            vol = None
            if npy.exists() and tz.exists():                     # reuse finished files; redo any cut off mid-write
                try:
                    vol = np.load(npy); tum = np.load(tz)["mask"]
                    if tum.shape != vol.shape: raise ValueError("shape mismatch")
                except Exception:
                    print(f"  {case}: saved files unreadable (interrupted run?), preparing again"); vol = None
            if vol is None:
                vol, tum = prepare_case(p, src / "labelsTr" / p.name)
                t1, t2 = npy.with_name(case + ".tmp.npy"), tz.with_name(case + ".tmp.npz")
                np.save(t1, vol); np.savez_compressed(t2, mask=tum); t1.replace(npy); t2.replace(tz)
            ts = np.flatnonzero(tum.sum((1, 2)) > 0)
            rows.append(dict(case=case, ok=True, shape=str(tuple(vol.shape)), n_slabs=vol.shape[0], tumor_slabs=len(ts),
                             tumor_voxels=int(tum.sum()), out_file=str(npy)))
        except Exception as e:
            rows.append(dict(case=case, ok=False, error=str(e)[:200])); print("FAILED", case, e)
        if i % 10 == 0 or i == len(imgs): print(f"{i}/{len(imgs)}", flush=True)
    r = pd.DataFrame(rows); f = ROOT / "data/processed/msd_preproc.csv"; f.parent.mkdir(parents=True, exist_ok=True); r.to_csv(f, index=False)
    print(f"wrote {f}: {int(r.ok.sum())} ok, {int((~r.ok).sum())} failed")
    if r.ok.any(): print(f"tumour slabs per scan: median {r[r.ok].tumor_slabs.median():.0f}, range {r[r.ok].tumor_slabs.min()}-{r[r.ok].tumor_slabs.max()}")


# ---------------------------------------------------------------- 3) colon masks, slab labels, split
def split_cases(cases, rng):
    c = np.array(sorted(cases)); rng.shuffle(c); n = len(c); ntr, nva = round(0.70 * n), round(0.15 * n)
    return {**{x: "train" for x in c[:ntr]}, **{x: "val" for x in c[ntr:ntr + nva]}, **{x: "test" for x in c[ntr + nva:]}}


def colon(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import select_colon as sc
    out = Path(a.root) / "processed"; mdir = out / "colon"; mdir.mkdir(parents=True, exist_ok=True)
    pre = pd.read_csv(ROOT / "data/processed/msd_preproc.csv"); pre = pre[pre.ok == True]
    if a.limit: pre = pre.head(a.limit)
    split = split_cases(pre.case, np.random.default_rng(SEED))
    segment = sc.totalseg_backend(a.device, a.fast)
    slab_rows, ser_rows = [], []
    for i, r in enumerate(pre.itertuples(), 1):
        mfile = mdir / f"{r.case}.npz"
        try:
            vol = np.load(r.out_file).astype(np.float32); tum = np.load(out / "tumor" / f"{r.case}.npz")["mask"]
            mask = None
            if mfile.exists():                                   # reuse a finished mask; redo one cut off mid-save
                try: mask = np.load(mfile)["mask"]
                except Exception: print(f"  {r.case}: saved mask unreadable (interrupted run?), segmenting again")
            if mask is None or mask.shape != vol.shape: mask = segment(vol).astype(bool)
            if mask.shape != vol.shape: raise ValueError(f"mask shape {mask.shape} != image shape {vol.shape}")
            rng, mask = sc.pick_range(mask, a.min_area, a.margin)
            tmp = mfile.with_name(mfile.stem + ".tmp.npz"); np.savez_compressed(tmp, mask=mask); tmp.replace(mfile)
            tpx = tum.sum((1, 2)); tz = np.flatnonzero(tpx > 0); n = vol.shape[0]
            if rng is None and len(tz) == 0: raise ValueError("no colon and no tumour found")
            starts = ([rng[0]] if rng else []) + ([int(tz.min())] if len(tz) else [])   # range always covers the whole tumour
            ends = ([rng[1]] if rng else []) + ([int(tz.max())] if len(tz) else [])
            lo, hi = min(starts), max(ends)
            area = mask.sum((1, 2)); counts = {"tumor": 0, "no_tumor": 0, "dropped": 0}
            for z in range(lo, hi + 1):
                dist = np.abs(tz - z).min() if len(tz) else 999
                lab = "tumor" if tpx[z] >= a.min_tumor_px else ("no_tumor" if tpx[z] == 0 and dist >= a.gap else "dropped")
                counts[lab] += 1
                if lab == "dropped": continue
                slab_rows.append(dict(patient_id=r.case, source="MSD", label=lab, split=split[r.case], series_uid=r.case,
                                      slab_idx=z, colon_area_px=int(area[z]), tumor_px=int(tpx[z]), n_slabs=n))
            ser_rows.append(dict(case=r.case, split=split[r.case], n_slabs=n, first_slab=lo, last_slab=hi,
                                 colon_found=rng is not None, status="ok", **{f"n_{k}": v for k, v in counts.items()}))
        except Exception as e:
            ser_rows.append(dict(case=r.case, split=split.get(r.case), status="error", error=str(e)[:150])); print("FAILED", r.case, e)
        if i % 10 == 0 or i == len(pre): print(f"{i}/{len(pre)}", flush=True)
    sl, ss = pd.DataFrame(slab_rows), pd.DataFrame(ser_rows); d = ROOT / "data/processed"
    sl.to_csv(d / "msd_colon_slices.csv", index=False); ss.to_csv(d / "msd_colon_series.csv", index=False)
    print(f"\nwrote {d / 'msd_colon_slices.csv'}: {len(sl)} slabs from {sl.patient_id.nunique() if len(sl) else 0} patients")
    if len(sl): print(pd.crosstab(sl.split, sl.label, margins=True).to_string())
    print(f"status: {ss.status.value_counts().to_dict()}")


# ---------------------------------------------------------------- 4) export for classify.py, 5) evaluate
LABEL_MAP = {"tumor": "malignant", "no_tumor": "benign"}       # classify.py's two labels; here: tumour slab / tumour-free slab


def export(a):
    """classify.py reads one label per patient from a split file.  Each scan becomes two pseudo-patients,
    '<case>__tumor' and '<case>__no_tumor', with the scan's split, so classify.py runs unchanged and a scan never
    crosses splits.  series_uid stays the case id, so preds.csv can be grouped back by scan."""
    d = ROOT / "data/processed"; s = pd.read_csv(d / "msd_colon_slices.csv")
    s["patient_id"] = s.series_uid + "__" + s.label; s["label"] = s.label.map(LABEL_MAP)
    s.to_csv(d / "msd_classify_slices.csv", index=False)
    sp = s.drop_duplicates("patient_id")[["patient_id", "split", "label"]]; sp.to_csv(d / "msd_classify_split.csv", index=False)
    print(f"wrote {d / 'msd_classify_slices.csv'} ({len(s)} slabs) and {d / 'msd_classify_split.csv'} ({len(sp)} pseudo-patients "
          f"from {s.series_uid.nunique()} scans)")
    print("train on Kaggle, e.g.\n  python src/classify.py --arch densenet121 --npy-dir <prepared scans folder> "
          "--slices data/processed/msd_classify_slices.csv --split data/processed/msd_classify_split.csv --out <run folder>")


def evaluate(a):
    from sklearn.metrics import roc_auc_score
    p = pd.read_csv(a.pred); p["case"] = p.series_uid; p["y"] = p.patient_id.str.endswith("__tumor").astype(int)
    rng = np.random.default_rng(SEED); rows = []
    for sp, g in p.groupby("split"):
        if g.y.nunique() < 2: continue
        y, s, c = g.y.values, g.score.values, g.case.values; auc = roc_auc_score(y, s)
        cases = np.unique(c); idx = {k: np.flatnonzero(c == k) for k in cases}; boot = []
        for _ in range(1000):                                    # bootstrap over scans, not slabs
            pick = np.concatenate([idx[k] for k in rng.choice(cases, len(cases))])
            if len(np.unique(y[pick])) == 2: boot.append(roc_auc_score(y[pick], s[pick]))
        within = [roc_auc_score(gg.y, gg.score) for _, gg in g.groupby("case") if gg.y.nunique() == 2]   # same scan, so no source effect
        pred = s >= 0.5
        rows.append(dict(name=a.name, split=sp, scans=len(cases), tumor_slabs=int(y.sum()), no_tumor_slabs=int((1 - y).sum()),
                         slab_auc=auc, auc_lo=np.percentile(boot, 2.5), auc_hi=np.percentile(boot, 97.5),
                         within_scan_auc_median=float(np.median(within)) if within else np.nan,
                         sensitivity=float(pred[y == 1].mean()), specificity=float((~pred[y == 0]).mean())))
    R = pd.DataFrame(rows); t = ROOT / "reports/tables/msd_eval.csv"; t.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(t) if t.exists() else pd.DataFrame()
    pd.concat([old[old.name != a.name] if len(old) else old, R], ignore_index=True).to_csv(t, index=False)
    pd.set_option("display.width", 200); print(R.round(3).to_string(index=False)); print("appended to", t)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("export")
    e = sub.add_parser("evaluate"); e.add_argument("--pred", required=True, help="preds.csv from classify.py"); e.add_argument("--name", required=True)
    for name in ("download", "prepare", "colon"):
        p = sub.add_parser(name); p.add_argument("--root", default="D:\\msd", help="folder on the big drive")
        p.add_argument("--limit", type=int, default=0, help="only the first N scans (to try it out)")
    sub.choices["download"].add_argument("--keep-tar", action="store_true")
    c = sub.choices["colon"]
    c.add_argument("--device", default="gpu"); c.add_argument("--fast", action="store_true")
    c.add_argument("--margin", type=int, default=1); c.add_argument("--min-area", type=int, default=150)
    c.add_argument("--min-tumor-px", type=int, default=20); c.add_argument("--gap", type=int, default=2)
    a = ap.parse_args()
    {"download": download, "prepare": prepare, "colon": colon, "export": export, "evaluate": evaluate}[a.step](a)


if __name__ == "__main__":
    main()
