"""Colon-only classifier: classify.py, but every pixel outside the automatic colon outline is set to air.

    python colon_only_classify.py --masks <folder with <series_uid>.npz> [--dilate 3] [--preview 12] <all classify.py options>

Why: a network that sees the whole slab can use body size, fat, bones or the table as a shortcut (the Grad-CAM maps
show most heat there).  Here the network sees only the colon (TotalSegmentator mask, grown by --dilate pixels so the
wall and the lumen edge are kept; 1 pixel = 1.5 mm, default 3 = 4.5 mm).  Everything else in the slab is air (-1).
This is an ADDITIONAL, clearly labelled experiment, not part of Table 1.  The colon contents can still differ between
the datasets (air-filled vs collapsed, contrast), so a high score is still not proof of tumour detection.

Everything else is identical to classify.py (same split, window, schedule, seeds, outputs; same --arch, --npy-dir,
--out, --encoder, --max-hours, ...).  The mask is applied BEFORE the small rotation/scale augmentation so they stay aligned.
--preview N   write <out>/colon_only_preview.png with N training slabs (mask applied) and stop; look at it first.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colon_dataset as cd

_CFG = {"masks": None, "dilate": 3}


_PACK = {}


def preload_masks(folder):
    """Read every mask once (bit-packed, about 1 GB for all 444 scans) before the data loader workers start,
    so the workers share it instead of decompressing a file for almost every slab."""
    for p in sorted(Path(folder).glob("*.npz")):
        m = np.load(p)["mask"].astype(bool); _PACK[p.stem] = (np.packbits(m, axis=-1), m.shape[-1])
    print(f"loaded {len(_PACK)} colon masks", flush=True)


def _mask(uid, slab):
    if uid not in _PACK: raise FileNotFoundError(f"no colon mask for scan {uid} in {_CFG['masks']}")
    pk, w = _PACK[uid]; return np.unpackbits(pk[slab], axis=-1, count=w).astype(bool)


def colon_only_item(self, i):
    r = self._r[i]; hu = np.asarray(cd._open(r["path"])[int(r["slab_idx"])], np.float32)
    m = _mask(r["series_uid"], int(r["slab_idx"]))
    if m.shape != hu.shape: raise ValueError(f"mask {m.shape} and scan {hu.shape} differ for {r['series_uid']} slab {r['slab_idx']}")
    img = cd._pad_square(cd._window(hu), self.size)
    mk = cd._pad_square(m.astype(np.float32) * 2 - 1, self.size) > 0
    if _CFG["dilate"] > 0: mk = binary_dilation(mk, iterations=_CFG["dilate"])
    img = np.where(mk, img, -1.0).astype(np.float32)               # outside the colon: air
    x = torch.from_numpy(img).unsqueeze(0)
    if self.augment: x = cd._augment(x)
    return dict(image=x.repeat(self.channels, 1, 1), label=torch.tensor(cd.LABELS[r["label"]]),
                patient_id=r["patient_id"], series_uid=r["series_uid"], slab_idx=int(r["slab_idx"]), source=r["source"])


def preview(a, n):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ds = cd.ColonSlices("train", a.npy_dir, a.slices, a.split, channels=3, size=a.size, augment=False)
    idx = np.linspace(0, len(ds) - 1, n).astype(int); cols = 4; rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows)); ax = np.atleast_1d(ax).ravel()
    for k, j in enumerate(idx):
        it = ds[int(j)]; ax[k].imshow(it["image"][0], cmap="gray", vmin=-1, vmax=1); ax[k].axis("off")
        ax[k].set_title(f"{it['source']} {it['patient_id'][-6:]} s{it['slab_idx']}", fontsize=7)
    for k in range(len(idx), len(ax)): ax[k].axis("off")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); f = out / "colon_only_preview.png"
    fig.tight_layout(); fig.savefig(f, dpi=110); print("wrote", f, "- check that the colon is inside the visible area and everything else is black")


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--masks", required=True); ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--preview", type=int, default=0)
    mine, rest = ap.parse_known_args()
    _CFG["masks"], _CFG["dilate"] = mine.masks, mine.dilate
    preload_masks(mine.masks)
    cd.ColonSlices.__getitem__ = colon_only_item                    # every dataset built by classify.py now masks
    sys.argv = [sys.argv[0]] + rest
    import classify
    if mine.preview:
        p = argparse.ArgumentParser(); p.add_argument("--npy-dir"); p.add_argument("--out"); p.add_argument("--slices", default=str(classify.ROOT / "data/processed/colon_slices.csv"))
        p.add_argument("--split", default=str(classify.ROOT / "data/processed/cohort_split.csv")); p.add_argument("--size", type=int, default=256)
        a, _ = p.parse_known_args(rest); preview(a, mine.preview); return
    print(f"colon-only: mask folder {mine.masks}, dilation {mine.dilate} px", flush=True)
    classify.main()


if __name__ == "__main__":
    main()
