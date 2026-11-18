"""Patient-level image features from a backbone, for the within-CT-Colonography test (within_ctc_cv.py --model embeddings).

    python src/extract_embeddings.py --arch resnet50 --encoder D:\\runs\\dcl_resnet_raw\\encoder.pt --npy-dir D:\\tcia\\processed --out D:\\runs\\emb_dcl_resnet.parquet
    python src/within_ctc_cv.py --model embeddings --embeddings D:\\runs\\emb_dcl_resnet.parquet --name dcl_resnet --npy-dir D:\\tcia\\processed

Why: inside CT Colonography every scan has the same preparation and protocol, so the source shortcut cannot work there.
Whether a backbone's features carry any signal about advanced adenoma versus other findings is therefore the fairest test of
what the network has learned.  Run it for three feature sources and compare:
    ImageNet weights only       (no --encoder, no --checkpoint)                  the starting point
    DenseCL encoder             --encoder <dcl run>/encoder.pt                    does label-free pretraining add anything?
    trained classifier          --checkpoint <classify run>/best.pt              features after benign-vs-malignant training
For the last one note that the classifier saw all CT Colonography training patients labelled 'benign', so it is the least clean;
ImageNet and DenseCL features never saw a label.

What it does: every labelled CT Colonography patient (train, val and test of the frozen split; the 3 cancers are left out), every
kept colon slab of every scan (or --max-slabs evenly spaced slabs per scan), same window and padding as the models, no augmentation.
Each slab -> global average pooled backbone features (1024 for DenseNet-121, 2048 for ResNet-50); averaged over a patient's slabs.
Output: parquet (or csv) with patient_id and f0, f1, ...
Needs: torch torchvision numpy pandas pyarrow (backbones.py, colon_dataset.py, classify.py in the same folder)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colon_dataset as cd
from backbones import Encoder, load_pretrained
from classify import Classifier

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["densenet121", "resnet50"], required=True)
    ap.add_argument("--npy-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default=None, help="DenseCL encoder.pt"); ap.add_argument("--checkpoint", default=None, help="classify.py best.pt")
    ap.add_argument("--imagenet-weights", default="imagenet", help="imagenet | none | path (used when neither --encoder nor --checkpoint is given)")
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--max-slabs", type=int, default=0, help="evenly spaced slabs per scan (0 = all; use 30 for a quick CPU run)")
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--size", type=int, default=256); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.encoder and a.checkpoint: ap.error("use --encoder or --checkpoint, not both")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
    if a.checkpoint:
        model = Classifier(Encoder(a.arch, "none")); model.load_state_dict(torch.load(a.checkpoint, map_location="cpu")); enc, src = model.enc, f"classifier {a.checkpoint}"
    elif a.encoder: enc, src = load_pretrained(a.arch, a.encoder), f"DenseCL {a.encoder}"
    else: enc, src = Encoder(a.arch, a.imagenet_weights), f"{a.imagenet_weights} weights"
    enc = enc.to(dev).eval()
    ds = cd.ColonSlices(["train", "val", "test"], a.npy_dir, a.slices, a.split, channels=3, size=a.size, augment=False, sources=["CTC"])
    R = ds.rows.reset_index(drop=True)
    if a.max_slabs:
        keep = []
        for _, g in R.groupby("series_uid"):
            g = g.sort_values("slab_idx"); idx = np.unique(np.round(np.linspace(0, len(g) - 1, min(a.max_slabs, len(g)))).astype(int)); keep += g.index[idx].tolist()
        ds = torch.utils.data.Subset(ds, sorted(keep)); R = R.loc[sorted(keep)].reset_index(drop=True)
    print(f"{a.arch} | {src} | {R.patient_id.nunique()} CT Colonography patients, {len(R)} slabs | device {dev.type}", flush=True)
    feats = []
    with torch.no_grad():
        for i, b in enumerate(DataLoader(ds, a.batch, shuffle=False, num_workers=a.workers)):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                f = enc(b["image"].to(dev)).float().mean((2, 3))
            feats.append(f.cpu().numpy())
            if i % 50 == 0: print(f"  batch {i}", flush=True)
    F = np.concatenate(feats)
    if not np.isfinite(F).all(): raise SystemExit("features contain NaN/inf: this encoder is broken, do not use it")
    cols = [f"f{i}" for i in range(F.shape[1])]; D = pd.DataFrame(F, columns=cols); D.insert(0, "series_uid", R.series_uid.values); D.insert(0, "patient_id", R.patient_id.values)
    P = D.drop(columns="series_uid").groupby("patient_id")[cols].mean().reset_index()       # mean over a patient's slabs and scans
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    P.to_parquet(a.out, index=False) if a.out.endswith(".parquet") else P.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(P)} patients x {len(cols)} features")


if __name__ == "__main__":
    main()
