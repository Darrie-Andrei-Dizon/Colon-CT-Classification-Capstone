"""Source-prediction check on the COLON region only.  Run it before and after CycleGAN.

    python src/source_check_colon.py --npy-dir D:\\tcia\\processed --label original
    python src/source_check_colon.py --npy-dir D:\\tcia\\translated_v1 --orig-dir D:\\tcia\\processed --label cyclegan_v1

Question it answers: can a simple model still tell which DATASET a scan came from (StageII or CT Colonography),
using only brightness numbers from the colon?  AUC near 1.0 = easy to tell (a shortcut exists);
AUC near 0.5 = cannot tell (the source difference is gone).  CycleGAN is judged by how far this falls.

Two regions per scan, because they answer different questions
  range  all body voxels in the kept colon slabs (what the models see, minus the air outside the body)
  mask   only the voxels inside the colon outline (the colon itself: gas, stool, contrast)
Two feature sets: 4 summaries (p1, p50, p99, share of voxels at the dark end of the window) and a 32-bin histogram.
All intensities are first clipped to the SAME window as the models (-400..400 HU), so original and translated
scans are compared on equal terms.  Geometry (slice thickness, pixel size) is never used.

Method: logistic regression on standardised features, trained on TRAIN patients, scored on TEST patients
(a patient's prone and supine scans stay together).  AUC 95% CI from a bootstrap over test patients.
Challenge and unlabelled patients are not used.

--orig-dir: where the ORIGINAL prepared scans are.  The body outline and colon masks always come from there, so
a translated folder (same file names, same shape, saved as float16 HU) is measured on exactly the same voxels.

Outputs: reports/tables/source_check_colon.csv (one block per --label, re-running a label replaces it),
         reports/figures/source_check_colon.png (all labels side by side)
Needs: numpy pandas scipy scikit-learn matplotlib
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
LO, HI = -400.0, 400.0                     # must match colon_dataset.WINDOW
EDGES = np.linspace(LO, HI, 33)
SEED = 20261023
N_BOOT = 1000


def body_mask(slabs):
    """Same recipe as preprocess.py: threshold, detach the table, largest piece, fill holes (colon gas stays inside)."""
    out = np.zeros(slabs.shape, bool)
    for i, s in enumerate(slabs):
        b = ndi.binary_opening(s > -300, iterations=3); lab, n = ndi.label(b)
        if n: b = lab == (1 + np.argmax(ndi.sum(b, lab, range(1, n + 1))))
        out[i] = ndi.binary_fill_holes(b)
    return out


def features(vals):
    v = np.clip(vals, LO, HI); p = np.percentile(v, [1, 50, 99])
    h, _ = np.histogram(v, bins=EDGES)
    d = dict(p1=p[0], p50=p[1], p99=p[2], dark_frac=float((v <= LO + 10).mean()))
    d.update({f"h{i:02d}": x for i, x in enumerate(h / max(h.sum(), 1))}); return d


def cluster_boot_auc(y, s, groups, rng):
    ug = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in ug}; out = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[pick])) == 2: out.append(roc_auc_score(y[pick], s[pick]))
    return np.percentile(out, [2.5, 97.5])


def load_slabs(path, z):
    """Read the chosen slabs; memory-mapping fails on some Windows drives (Errno 22), so fall back to reading the whole file."""
    try: return np.load(path, mmap_mode="r")[z].astype(np.float32)
    except OSError: return np.load(path)[z].astype(np.float32)


def build_features(a):
    sl = pd.read_csv(a.slices); cs = pd.read_csv(a.split)[["patient_id", "split"]]
    sl = sl.drop(columns=["split"]).merge(cs, on="patient_id")
    sl = sl[sl.split.isin(["train", "test"])]
    rows = []; n = sl.series_uid.nunique(); mdir = Path(a.masks)
    for k, (uid, g) in enumerate(sl.groupby("series_uid"), 1):
        z = np.sort(g.slab_idx.values)
        try:
            orig = load_slabs(Path(a.orig_dir) / f"{uid}.npy", z)
            tp = Path(a.npy_dir) / f"{uid}.npy"
            same = Path(a.npy_dir).resolve() == Path(a.orig_dir).resolve() or not tp.exists()   # scans not translated (CT Colonography) stay as they are
            cur = orig if same else load_slabs(tp, z)
            if cur.shape != orig.shape: raise ValueError(f"shape {cur.shape} != original {orig.shape}")
            body = body_mask(orig); colon = np.load(mdir / f"{uid}.npz")["mask"][z]
            for region, m in (("range", body), ("mask", colon & body)):
                if m.sum() < 500: print(f"skip {region} for {uid[-10:]}: only {int(m.sum())} voxels"); continue
                rows.append(dict(series_uid=uid, patient_id=g.patient_id.iloc[0], source=g.source.iloc[0],
                                 split=g.split.iloc[0], region=region, **features(cur[m])))
        except Exception as e:
            print("FAILED", uid[-10:], e)
        if k % 25 == 0 or k == n: print(f"{k}/{n}", flush=True)
    return pd.DataFrame(rows)


def run_models(F, label):
    sets = {"4 summaries": ["p1", "p50", "p99", "dark_frac"], "32-bin histogram": [f"h{i:02d}" for i in range(32)]}
    rng = np.random.default_rng(SEED); res = []
    for region in ["range", "mask"]:
        for name, cols in sets.items():
            d = F[F.region == region]; tr, te = d[d.split == "train"], d[d.split == "test"]
            y_tr, y_te = (tr.source == "StageII").astype(int).values, (te.source == "StageII").astype(int).values
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2: print(f"skip {region}/{name}: one class missing"); continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(tr[cols], y_tr)
            s = m.predict_proba(te[cols])[:, 1]; lo, hi = cluster_boot_auc(y_te, s, te.patient_id.values, rng)
            pred = (s >= 0.5).astype(int)
            bal = (((pred == 1) & (y_te == 1)).sum() / (y_te == 1).sum() + ((pred == 0) & (y_te == 0)).sum() / (y_te == 0).sum()) / 2
            res.append(dict(label=label, region=region, features=name, auc=roc_auc_score(y_te, s), auc_lo=lo, auc_hi=hi,
                            balanced_accuracy=bal, n_train=len(tr), n_test=len(te), n_test_patients=te.patient_id.nunique()))
    return pd.DataFrame(res)


def plot(R, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labels = list(dict.fromkeys(R.label)); cats = list(dict.fromkeys(zip(R.region, R.features)))
    cols = ["#2e86e4", "#e4572e", "#2ea36a", "#8a5ad6", "#c9a227"]
    fig, ax = plt.subplots(figsize=(1.8 * len(cats) + 2, 4.4)); w = 0.8 / len(labels)
    for j, lab in enumerate(labels):
        for i, c in enumerate(cats):
            r = R[(R.label == lab) & (R.region == c[0]) & (R.features == c[1])]
            if r.empty: continue
            r = r.iloc[0]; x = i + (j - (len(labels) - 1) / 2) * w
            ax.bar(x, r.auc, w * 0.9, color=cols[j % len(cols)], label=lab if i == 0 else None)
            ax.errorbar(x, r.auc, yerr=[[r.auc - r.auc_lo], [r.auc_hi - r.auc]], color="#333", capsize=2, lw=1)
            ax.text(x, min(r.auc_hi + 0.02, 1.07), f"{r.auc:.2f}", ha="center", fontsize=7)
    ax.axhline(0.5, ls="--", color="#999", lw=1); ax.text(-0.48, 0.41, "dashed line = 0.5, cannot tell", ha="left", fontsize=7, color="#777")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([f"{a}\n{b}" for a, b in cats], fontsize=8)
    ax.set_ylim(0.4, 1.1); ax.set_ylabel("AUC: telling StageII from CT Colonography")
    ax.set_title("Can a simple model still tell the source? (colon region, test patients)", fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir", required=True, help="scans to measure (original or translated)")
    ap.add_argument("--orig-dir", help="original prepared scans (body outline + colon masks); default = --npy-dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--masks", default=str(ROOT / "data/processed/colon"))
    ap.add_argument("--table", default=str(ROOT / "reports/tables/source_check_colon.csv"))
    ap.add_argument("--fig", default=str(ROOT / "reports/figures/source_check_colon.png"))
    a = ap.parse_args(); a.orig_dir = a.orig_dir or a.npy_dir

    F = build_features(a)
    if F.empty or F.source.nunique() < 2: raise SystemExit("need scans from both StageII and CT Colonography; check --npy-dir / --orig-dir")
    res = run_models(F, a.label)
    t = Path(a.table); t.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(t) if t.exists() else pd.DataFrame()
    allr = pd.concat([old[old.label != a.label] if len(old) else old, res], ignore_index=True); allr.to_csv(t, index=False)
    pd.set_option("display.width", 200)
    print(f"\n{a.label}: StageII vs CT Colonography, test patients")
    print(res.drop(columns="label").round(3).to_string(index=False))
    try: plot(allr, a.fig); print("figure:", a.fig)
    except Exception as e: print("figure not saved:", e, "(try --fig with another folder)")


if __name__ == "__main__":
    main()
