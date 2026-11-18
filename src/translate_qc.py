"""Quality check of CycleGAN-translated scans (manuscript 3.5, stage 2): did the translation keep the anatomy?

    python src/translate_qc.py --orig-dir D:\\tcia\\processed --trans-dir D:\\tcia\\translated_v1 --label cyclegan_v1

For every kept colon slab of every scan that was actually changed by the translation it measures
  iou          overlap of the body outline before and after (1.00 = same outline)
  area_ratio   body area after / before (1.00 = same size; a bigger or smaller body means the generator changed the geometry)
  ssim         structural similarity inside the body, on the model window (-400..400 HU)
  colon_mad    mean absolute HU change inside the colon outline (how much the colon itself was rewritten)
  gas_before / gas_after   share of body voxels at the dark end of the window (gas): shows gas being added or removed
A slab is REJECTED when  iou < --iou-min  or  |area_ratio - 1| > --area-tol  or  ssim < --ssim-min.
The thresholds are provisional.  Look at the printed distribution and the worst-slab picture, then FIX them before looking at
any classification result and write the final values in the paper (the manuscript requires a stated rejection threshold).

Why these checks: in the sample pictures the generator sometimes enlarged the body or painted gas over soft tissue.
The source re-test (source_check_colon.py) uses the ORIGINAL body and colon outlines, so a slab whose body changed size is
measured on the wrong voxels; rejected slabs should be left out of that test and out of training.

Outputs
  reports/tables/translate_qc_<label>_slabs.csv    one row per changed slab, with all measures and 'reject'
  reports/tables/translate_qc_<label>_series.csv   one row per scan (means, worst slab, share rejected)
  data/processed/colon_slices_<label>.csv          colon_slices.csv without the rejected slabs (turn off with --no-filtered)
  reports/figures/translate_qc_<label>_hist.png    distributions with the thresholds
  reports/figures/translate_qc_<label>_worst.png   the worst slabs: before | after | difference
Scans the translation did not change (the other domain is copied as it was) are skipped and listed as 'unchanged'.
Needs: numpy pandas scipy matplotlib
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
LO, HI = -400.0, 400.0                                    # must match colon_dataset.WINDOW


def body_mask(s):
    """Same recipe as preprocess.py / source_check_colon.py, for one slab."""
    b = ndi.binary_opening(s > -300, iterations=3); lab, n = ndi.label(b)
    if n: b = lab == (1 + np.argmax(ndi.sum(b, lab, range(1, n + 1))))
    return ndi.binary_fill_holes(b)


def window01(hu):
    return (np.clip(hu, LO, HI) - LO) / (HI - LO)


def ssim_map(a, b, win=7):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    f = lambda x: ndi.uniform_filter(x, win)
    ma, mb = f(a), f(b); va, vb, cab = f(a * a) - ma * ma, f(b * b) - mb * mb, f(a * b) - ma * mb
    return ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))


def slab_metrics(o, t, colon):
    bo, bt = body_mask(o), body_mask(t); union, inter = bo | bt, bo & bt
    iou = inter.sum() / max(union.sum(), 1); ratio = bt.sum() / max(bo.sum(), 1)
    wo, wt = window01(o), window01(t); sm = ssim_map(wo, wt)
    m = union if union.sum() > 50 else np.ones_like(union)
    cm = colon & bo
    return dict(iou=float(iou), area_ratio=float(ratio), ssim=float(sm[m].mean()),
                colon_mad=float(np.abs(o - t)[cm].mean()) if cm.sum() > 50 else np.nan,
                gas_before=float((o[bo] <= LO + 10).mean()) if bo.any() else np.nan,
                gas_after=float((t[bt] <= LO + 10).mean()) if bt.any() else np.nan)


def load_slabs(path, z):
    """Read the chosen slabs; memory-mapping fails on some Windows drives (Errno 22), so fall back to reading the whole file."""
    try: return np.load(path, mmap_mode="r")[z].astype(np.float32)
    except OSError: return np.load(path)[z].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True); ap.add_argument("--trans-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--masks", default=str(ROOT / "data/processed/colon"))
    ap.add_argument("--iou-min", type=float, default=0.90); ap.add_argument("--area-tol", type=float, default=0.10)
    ap.add_argument("--ssim-min", type=float, default=0.70)
    ap.add_argument("--tables", default=str(ROOT / "reports/tables")); ap.add_argument("--fig-dir", default=str(ROOT / "reports/figures"))
    ap.add_argument("--filtered", default=None, help="path for the filtered slices csv (default data/processed/colon_slices_<label>.csv)")
    ap.add_argument("--no-filtered", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sl = pd.read_csv(a.slices); cs = pd.read_csv(a.split)[["patient_id", "split"]]
    sl = sl.drop(columns=["split"]).merge(cs, on="patient_id", how="left")
    uids = list(dict.fromkeys(sl.series_uid)); uids = uids[:a.limit] if a.limit else uids
    rows, unchanged, failed, keep_img, rand_img = [], [], [], {}, {}
    rng = np.random.default_rng(20261106)
    for k, uid in enumerate(uids, 1):
        g = sl[sl.series_uid == uid]; z = np.sort(g.slab_idx.values)
        try:
            tp = Path(a.trans_dir) / f"{uid}.npy"
            if not tp.exists(): unchanged.append(uid); continue          # not translated (e.g. CT Colonography scans): nothing to check
            O = load_slabs(Path(a.orig_dir) / f"{uid}.npy", z); T = load_slabs(tp, z)
            if O.shape != T.shape: raise ValueError(f"shape {T.shape} != {O.shape}")
            if np.array_equal(O, T): unchanged.append(uid); continue
            colon = np.load(Path(a.masks) / f"{uid}.npz")["mask"][z]
            ri = int(rng.integers(len(z)))                       # one random slab per scan; every 9th scan is kept for the 'typical slabs' figure
            for i, s in enumerate(z):
                d = slab_metrics(O[i], T[i], colon[i])
                d.update(series_uid=uid, patient_id=g.patient_id.iloc[0], source=g.source.iloc[0], split=g.split.iloc[0],
                         label=g.label.iloc[0], slab_idx=int(s)); rows.append(d)
                keep_img[(uid, int(s))] = (d["ssim"], O[i].copy(), T[i].copy())
                if i == ri and k % 9 == 0: rand_img[(uid, int(s))] = (d, O[i].copy(), T[i].copy())
                if len(keep_img) > 60:                                    # memory: keep only the 40 lowest-SSIM slabs
                    for q in sorted(keep_img, key=lambda q: keep_img[q][0])[40:]: keep_img.pop(q)
        except Exception as e:
            failed.append(uid); print("FAILED", uid[-10:], e)
        if k % 20 == 0 or k == len(uids): print(f"{k}/{len(uids)}", flush=True)
    if not rows: print("no changed scans found - nothing to check"); return
    S = pd.DataFrame(rows)
    S["reject"] = (S.iou < a.iou_min) | ((S.area_ratio - 1).abs() > a.area_tol) | (S.ssim < a.ssim_min)
    S["why"] = [";".join(n for n, c in (("outline", r.iou < a.iou_min), ("size", abs(r.area_ratio - 1) > a.area_tol), ("ssim", r.ssim < a.ssim_min)) if c)
                for r in S.itertuples()]
    tdir = Path(a.tables); tdir.mkdir(parents=True, exist_ok=True)
    S.to_csv(tdir / f"translate_qc_{a.label}_slabs.csv", index=False)
    P = S.groupby("series_uid").agg(patient_id=("patient_id", "first"), source=("source", "first"), split=("split", "first"),
                                    label=("label", "first"), n_slabs=("slab_idx", "size"), iou_mean=("iou", "mean"), iou_min=("iou", "min"),
                                    area_ratio_mean=("area_ratio", "mean"), ssim_mean=("ssim", "mean"), ssim_min=("ssim", "min"),
                                    colon_mad_mean=("colon_mad", "mean"), gas_before=("gas_before", "mean"), gas_after=("gas_after", "mean"),
                                    share_rejected=("reject", "mean")).reset_index()
    P.to_csv(tdir / f"translate_qc_{a.label}_series.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nchanged scans: {P.shape[0]}  | unchanged (other domain): {len(unchanged)}  | failed: {len(failed)}")
    print("distribution over changed slabs (percentiles 5 / 25 / 50 / 75 / 95):")
    for c in ["iou", "area_ratio", "ssim", "colon_mad", "gas_before", "gas_after"]:
        q = np.nanpercentile(S[c], [5, 25, 50, 75, 95]); print(f"  {c:11s}", "  ".join(f"{v:8.3f}" for v in q))
    print(f"thresholds: iou >= {a.iou_min}, |area_ratio - 1| <= {a.area_tol}, ssim >= {a.ssim_min}   (PROVISIONAL, fix before seeing results)")
    print(f"slabs rejected: {int(S.reject.sum())} of {len(S)} ({S.reject.mean():.1%}); by reason: {S[S.reject].why.value_counts().to_dict()}")
    print(f"scans with more than half their slabs rejected: {int((P.share_rejected > 0.5).sum())} of {len(P)}")
    print("by split (share of slabs rejected):", S.groupby("split").reject.mean().round(3).to_dict())

    if not a.no_filtered:
        bad = set(zip(S[S.reject].series_uid, S[S.reject].slab_idx))
        full = pd.read_csv(a.slices); keep = [(u, int(s)) not in bad for u, s in zip(full.series_uid, full.slab_idx)]
        fp = Path(a.filtered or ROOT / f"data/processed/colon_slices_{a.label}.csv"); fp.parent.mkdir(parents=True, exist_ok=True)
        full[keep].to_csv(fp, index=False); print(f"filtered slice list: {fp}  ({sum(keep)} of {len(full)} slabs kept)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fdir = Path(a.fig_dir); fdir.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, (c, thr, ttl) in zip(axs, [("iou", a.iou_min, "body outline overlap"), ("area_ratio", None, "body size after / before"), ("ssim", a.ssim_min, "SSIM inside body")]):
            ax.hist(S[c].dropna(), bins=50, color="#2e86e4"); ax.set_title(ttl, fontsize=9)
            if thr is not None: ax.axvline(thr, color="#e4572e", ls="--")
            if c == "area_ratio": [ax.axvline(v, color="#e4572e", ls="--") for v in (1 - a.area_tol, 1 + a.area_tol)]
        fig.tight_layout(); fig.savefig(fdir / f"translate_qc_{a.label}_hist.png", dpi=130); plt.close(fig)
        worst = S.sort_values("ssim").head(8); items = [(r.series_uid, r.slab_idx, r) for r in worst.itertuples() if (r.series_uid, r.slab_idx) in keep_img]
        if items:
            fig, axs = plt.subplots(len(items), 3, figsize=(7.5, 2.5 * len(items)), squeeze=False)
            for row, (u, s, r) in zip(axs, items):
                _, o, t = keep_img[(u, s)]; row[0].imshow(o, cmap="gray", vmin=LO, vmax=HI); row[1].imshow(t, cmap="gray", vmin=LO, vmax=HI)
                row[2].imshow(np.abs(o - t), cmap="magma", vmin=0, vmax=400)
                row[0].set_title(f"{r.source} slab {s} before", fontsize=7); row[1].set_title(f"after  ssim {r.ssim:.2f} iou {r.iou:.2f} size {r.area_ratio:.2f}", fontsize=7)
                row[2].set_title("difference (HU)", fontsize=7); [x.axis("off") for x in row]
            fig.tight_layout(); fig.savefig(fdir / f"translate_qc_{a.label}_worst.png", dpi=110); plt.close(fig)
        if rand_img:                                        # typical slabs: random scans, not the worst ones
            ks = list(rand_img); pick = [ks[i] for i in rng.permutation(len(ks))[:10]]
            fig, axs = plt.subplots(len(pick), 3, figsize=(7.5, 2.5 * len(pick)), squeeze=False)
            for row, key in zip(axs, pick):
                d, o, t = rand_img[key]; row[0].imshow(o, cmap="gray", vmin=LO, vmax=HI); row[1].imshow(t, cmap="gray", vmin=LO, vmax=HI)
                row[2].imshow(np.abs(o - t), cmap="magma", vmin=0, vmax=400)
                row[0].set_title(f"{d['source']} slab {key[1]} before", fontsize=7); row[1].set_title(f"after  ssim {d['ssim']:.2f} iou {d['iou']:.2f} size {d['area_ratio']:.2f}", fontsize=7)
                row[2].set_title("difference (HU)", fontsize=7); [x.axis("off") for x in row]
            fig.tight_layout(); fig.savefig(fdir / f"translate_qc_{a.label}_random.png", dpi=110); plt.close(fig)
        print("figures:", fdir)
    except Exception as e:
        print("figures not saved:", e, "(try --fig-dir with another folder)")


if __name__ == "__main__":
    main()
