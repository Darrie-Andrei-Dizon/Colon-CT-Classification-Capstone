"""One shared data loader for CycleGAN, DenseCL and the DenseNet-121 / ResNet-50 classifiers.

    from colon_dataset import ColonSlices, UnpairedColon, seed_everything, seed_worker, predictions_frame

Why it exists: every model must see the SAME images, with the SAME split and the SAME intensity window.
It reads only the frozen files:  cohort_split.csv (split + label), colon_slices.csv (which slabs), and the
prepared scans <npy_dir>/<series_uid>.npy made by preprocess.py.  Nothing is re-split and no DICOM is read.

Image recipe (fixed, do not change after training starts)
  prepared scan (HU, 1.5 mm in-plane, 5 mm slab) -> clip to WINDOW = [-400, 400] HU -> scale to [-1, 1]
  -> centre-pad (with air) to size x size, NO resizing, so the pixel size stays 1.5 mm.
  Output tensor: float32 (channels, size, size); channels = 1 (CycleGAN) or 3 (pretrained DenseNet / ResNet).

Leakage guards
  * UnpairedColon (CycleGAN / DenseCL training) only accepts splits "train" and "pretrain"; asking for
    val, test or challenge raises an error.
  * The split in colon_slices.csv is checked against cohort_split.csv; a mismatch stops the run.
  * Augmentation is small rotation / scale only.  There is NO left-right flip (colon anatomy has a fixed side).

Translated scans: save CycleGAN output as float16 HU (use to_hu) in a new folder with the SAME file names
(<series_uid>.npy, same shape), then pass that folder as npy_dir.  The classifiers and the source re-test then
read it exactly like the original.

Check it works:   python src/colon_dataset.py --check --npy-dir D:\\tcia\\processed
Needs: pip install torch numpy pandas
"""
import argparse, random
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
WINDOW = (-400.0, 400.0)          # HU.  Chosen once for the whole pipeline.
SEED = 20261023
TRAIN_SIDE = {"train", "pretrain"}
LABELS = {"benign": 0, "malignant": 1}


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):                       # DataLoader(..., worker_init_fn=seed_worker)
    s = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(s); random.seed(s)


def to_hu(x):
    """Inverse of the window: [-1, 1] -> HU in [-400, 400].  Use when saving CycleGAN output."""
    return x * (WINDOW[1] - WINDOW[0]) / 2 + (WINDOW[1] + WINDOW[0]) / 2


def _window(hu):
    lo, hi = WINDOW
    return (np.clip(hu, lo, hi) - lo) / (hi - lo) * 2 - 1


@lru_cache(maxsize=16)
def _open(path):
    return np.load(path, mmap_mode="r")


def _pad_square(img, size):
    """Centre-pad with air (-1) / centre-crop to size x size, no resampling."""
    h, w = img.shape; out = np.full((size, size), -1.0, np.float32)
    ys, xs = max((h - size) // 2, 0), max((w - size) // 2, 0); ch, cw = min(h, size), min(w, size)
    oy, ox = (size - ch) // 2, (size - cw) // 2
    out[oy:oy + ch, ox:ox + cw] = img[ys:ys + ch, xs:xs + cw]; return out


def _augment(x, max_deg=10.0, scale=(0.95, 1.05)):
    """Small random rotation + scale on a (C, H, W) tensor.  Air stays air (zero-fill = -1 via shift)."""
    ang = float(np.random.uniform(-max_deg, max_deg)) * np.pi / 180; s = float(np.random.uniform(*scale))
    c, sn = np.cos(ang) / s, np.sin(ang) / s
    theta = torch.tensor([[c, -sn, 0.0], [sn, c, 0.0]], dtype=torch.float32).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(theta, (1,) + tuple(x.shape), align_corners=False)
    y = torch.nn.functional.grid_sample(x.unsqueeze(0) + 1, grid, padding_mode="zeros", align_corners=False)
    return y.squeeze(0) - 1


def load_rows(slices_csv, split_csv, npy_dir, splits, sources=None, labels=None):
    rows = pd.read_csv(slices_csv); cs = pd.read_csv(split_csv)[["patient_id", "split", "label"]]
    m = rows.drop(columns=["split", "label"]).merge(cs, on="patient_id", how="left", validate="m:1")
    if m.split.isna().any(): raise ValueError("some patients in colon_slices.csv are not in cohort_split.csv")
    if (m.split.values != rows.split.values).any():
        raise ValueError("split in colon_slices.csv disagrees with cohort_split.csv - regenerate colon_slices.csv")
    keep = m.split.isin(splits)
    if sources: keep &= m.source.isin(sources)
    if labels: keep &= m.label.isin(labels)
    m = m[keep].reset_index(drop=True)
    m["path"] = [str(Path(npy_dir) / f"{u}.npy") for u in m.series_uid]
    return m


class ColonSlices(Dataset):
    """Labelled colon slabs for one split.  Item: dict(image, label, patient_id, series_uid, slab_idx, source)."""

    def __init__(self, split, npy_dir, slices_csv=None, split_csv=None, channels=1, size=256, augment=False,
                 sources=None):
        splits = [split] if isinstance(split, str) else list(split)
        self.rows = load_rows(slices_csv or ROOT / "data/processed/colon_slices.csv",
                              split_csv or ROOT / "data/processed/cohort_split.csv", npy_dir, splits, sources,
                              labels=list(LABELS))
        if not len(self.rows): raise ValueError(f"no labelled slabs for split={splits}")
        self.channels, self.size, self.augment = channels, size, augment
        self._r = self.rows.to_dict("records")

    def __len__(self): return len(self._r)

    def __getitem__(self, i):
        r = self._r[i]; hu = np.asarray(_open(r["path"])[int(r["slab_idx"])], np.float32)
        x = torch.from_numpy(_pad_square(_window(hu), self.size)).unsqueeze(0)
        if self.augment: x = _augment(x)
        return dict(image=x.repeat(self.channels, 1, 1), label=torch.tensor(LABELS[r["label"]]),
                    patient_id=r["patient_id"], series_uid=r["series_uid"], slab_idx=int(r["slab_idx"]),
                    source=r["source"])

    def slice_weights(self):
        """Weight so every PATIENT counts equally (a patient with 80 slabs does not outweigh one with 30).
        Use with torch.utils.data.WeightedRandomSampler(weights, len(ds), replacement=True)."""
        n = self.rows.groupby("patient_id").slab_idx.transform("size").values
        return torch.tensor(1.0 / n, dtype=torch.double)


class UnpairedColon(Dataset):
    """For CycleGAN: domain A = StageII slabs, domain B = CT Colonography slabs, train side only, unpaired.
    Item: dict(A, B, ...).  B is drawn at random (seeded) so pairing changes every epoch.
    Which direction to translate (A->B or B->A) is a project decision, not made here."""

    def __init__(self, npy_dir, splits=("train",), slices_csv=None, split_csv=None, channels=1, size=256,
                 augment=False, a_source="StageII", b_source="CTC"):
        bad = set(splits) - TRAIN_SIDE
        if bad: raise ValueError(f"{sorted(bad)} cannot be used to train CycleGAN / DenseCL (train side only: {sorted(TRAIN_SIDE)})")
        sc = slices_csv or ROOT / "data/processed/colon_slices.csv"; sp = split_csv or ROOT / "data/processed/cohort_split.csv"
        self.A = ColonSlices.__new__(ColonSlices); self.B = ColonSlices.__new__(ColonSlices)
        for ds, src in ((self.A, a_source), (self.B, b_source)):
            ds.rows = load_rows(sc, sp, npy_dir, list(splits), [src]); ds.channels, ds.size, ds.augment = channels, size, augment
            ds._r = ds.rows.to_dict("records")
            if not len(ds.rows): raise ValueError(f"no slabs for source {src} in splits {splits}")
            ds.rows["label"] = ds.rows.label.fillna("")
        self.rng = np.random.default_rng(SEED)

    def __len__(self): return max(len(self.A), len(self.B))

    def _get(self, ds, i):
        r = ds._r[i]; hu = np.asarray(_open(r["path"])[int(r["slab_idx"])], np.float32)
        x = torch.from_numpy(_pad_square(_window(hu), ds.size)).unsqueeze(0)
        if ds.augment: x = _augment(x)
        return x.repeat(ds.channels, 1, 1), r

    def __getitem__(self, i):
        a, ra = self._get(self.A, i % len(self.A)); b, rb = self._get(self.B, int(np.random.randint(len(self.B))))
        return dict(A=a, B=b, A_patient=ra["patient_id"], B_patient=rb["patient_id"])


def predictions_frame(ds, scores):
    """Turn model scores (P(malignant), one per dataset item in order) into the CSV evaluate.py expects."""
    s = np.asarray(scores, float)
    if len(s) != len(ds): raise ValueError(f"{len(s)} scores for {len(ds)} slabs")
    return pd.DataFrame(dict(patient_id=ds.rows.patient_id.values, series_uid=ds.rows.series_uid.values,
                             slab_idx=ds.rows.slab_idx.values, score=s))


def _check(a):
    seed_everything()
    for sp in ["train", "val", "test"]:
        try: ds = ColonSlices(sp, a.npy_dir, a.slices, a.split)
        except ValueError as e: print(f"{sp:5s} {e}"); continue
        n = ds.rows.groupby("label").patient_id.nunique().to_dict(); k = ds.rows.groupby("label").size().to_dict()
        print(f"{sp:5s} slabs {k}   patients {n}")
    allr = load_rows(a.slices or ROOT / "data/processed/colon_slices.csv", a.split or ROOT / "data/processed/cohort_split.csv",
                     a.npy_dir, ["train", "val", "test", "challenge"])
    share = allr.groupby("patient_id").split.nunique()
    assert (share == 1).all(), "a patient appears in more than one split"
    print("each patient is in exactly one split: ok")
    missing = [p for p in allr.path.unique() if not Path(p).exists()]
    print(f"prepared scan files: {len(allr.path.unique()) - len(missing)} found, {len(missing)} missing")
    ds = ColonSlices("train", a.npy_dir, a.slices, a.split, channels=3, augment=True); x = ds[0]["image"]
    print("one item:", tuple(x.shape), x.dtype, f"range [{x.min():.2f}, {x.max():.2f}]")
    try:
        UnpairedColon(a.npy_dir, splits=("val",)); print("ERROR: val was accepted for CycleGAN")
    except ValueError as e: print("guard ok:", str(e)[:70], "...")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ds = ColonSlices("train", a.npy_dir, a.slices, a.split); idx = np.random.default_rng(SEED).choice(len(ds), 8, replace=False)
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for ax, i in zip(axs.ravel(), idx):
        it = ds[int(i)]; ax.imshow(it["image"][0], cmap="gray", vmin=-1, vmax=1); ax.axis("off")
        ax.set_title(f"{it['source']} {ds.rows.label[int(i)]} slab {it['slab_idx']}", fontsize=8)
    out = ROOT / "reports/figures/colon_dataset_check.png"; out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=110); print("sample image:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ap.add_argument("--npy-dir", required=True); ap.add_argument("--slices"); ap.add_argument("--split")
    a = ap.parse_args()
    if a.check: _check(a)
