"""Grad-CAM heat maps for a trained classifier (manuscript 3.5 stage 8): where in the slab does the model look?

    python gradcam.py --arch densenet121 --checkpoint <run>/best.pt --npy-dir <scans> --slices colon_slices.csv \\
        --split cohort_split.csv --masks data/processed/colon --out <run>/gradcam [--cases panel_cases.csv] [--split-name test]

Method: Grad-CAM (Selvaraju et al. 2017) on the encoder's last feature map (DenseNet: after norm5 + ReLU; ResNet: layer4).
Weights = mean gradient of the malignant logit over each channel; map = ReLU(sum weights * channels), upsampled to the image
size and scaled to 0-1 per slab.  Everything is run in eval mode, no augmentation.

Which slabs: --cases csv (columns patient_id, optionally series_uid and slab_idx; e.g. the panel cases from select_panel_cases.py);
otherwise one slab per patient in --split-name (default test): the slab with the highest model score if --preds is given,
otherwise the slab with the largest colon outline.  --max-patients limits how many.

Also measured per slab (needs --masks): cam_in_colon = share of heat-map weight that falls inside the colon outline, next to
colon_frac = the share of the image the colon covers.  If cam_in_colon is not clearly above colon_frac, the model is looking
outside the colon (body outline, table, fat), which is the source-shortcut warning sign.  Heat maps highlight where the
model looked, not proof that it found a lesion; the clinician panel judges that.

Outputs: <out>/<patient_id>_<slab>.png (CT | heat map overlay), <out>/montage.png, <out>/gradcam_summary.csv, <out>/maps.npz
Needs: torch torchvision numpy pandas matplotlib (classify.py, backbones.py, colon_dataset.py in the same folder)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colon_dataset as cd
from backbones import Encoder
from classify import Classifier

ROOT = Path(__file__).resolve().parent.parent


def grad_cam(model, x):
    """x: (1,3,H,W) in [-1,1].  Returns (map (H,W) in 0-1, logit)."""
    model.eval(); x = x.clone().requires_grad_(True)
    feat = model.enc(x); feat.retain_grad()
    logit = model.fc(model.drop(feat.mean((2, 3)))).squeeze()
    model.zero_grad(); logit.backward()
    w = feat.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * feat).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
    cam = cam - cam.min(); cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.numpy(), float(logit.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["densenet121", "resnet50"], required=True)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--npy-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--masks", default=str(ROOT / "data/processed/colon"))
    ap.add_argument("--cases", default=None); ap.add_argument("--preds", default=None)
    ap.add_argument("--split-name", default="test", choices=["train", "val", "test", "challenge"])
    ap.add_argument("--max-patients", type=int, default=24); ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = Classifier(Encoder(a.arch, "none")).to(dev); model.load_state_dict(torch.load(a.checkpoint, map_location=dev)); model.eval()
    ds = cd.ColonSlices(a.split_name if not a.cases else ["train", "val", "test", "challenge"], a.npy_dir, a.slices, a.split, channels=3, size=a.size)
    R = ds.rows.reset_index(drop=True); R["i"] = np.arange(len(R))
    if a.cases:
        cs = pd.read_csv(a.cases); picks = []
        for r in cs.itertuples():
            g = R[R.patient_id == r.patient_id]
            if hasattr(r, "series_uid") and isinstance(getattr(r, "series_uid"), str):
                g = g[(g.series_uid == r.series_uid) & (g.slab_idx == int(r.slab_idx))]
            else: g = g.sort_values("colon_area_px", ascending=False).head(1)
            if len(g): picks.append(int(g.i.iloc[0]))
    else:
        if a.preds:
            pr = pd.read_csv(a.preds); R = R.merge(pr[["patient_id", "series_uid", "slab_idx", "score"]], on=["patient_id", "series_uid", "slab_idx"], how="left")
            g = R.sort_values("score", ascending=False).groupby("patient_id").head(1)
        else:
            g = R.sort_values("colon_area_px", ascending=False).groupby("patient_id").head(1)
        picks = g.sort_values("patient_id").i.tolist()[:a.max_patients]
    print(f"{len(picks)} slabs; model {a.arch}; device {dev.type}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows, maps, shown = [], {}, []
    for i in picks:
        item = ds[i]; x = item["image"].unsqueeze(0).to(dev); cam, logit = grad_cam(model, x)
        p = 1 / (1 + np.exp(-logit)); img = item["image"][0].numpy(); uid, s = item["series_uid"], item["slab_idx"]
        d = dict(patient_id=item["patient_id"], series_uid=uid, slab_idx=s, label="malignant" if int(item["label"]) else "benign", score=p)
        mp = Path(a.masks) / f"{uid}.npz"
        if mp.exists():
            m = cd._pad_square(np.load(mp)["mask"][s].astype(np.float32) * 2 - 1, a.size) > 0
            d["colon_frac"] = float(m.mean()); d["cam_in_colon"] = float(cam[m].sum() / max(cam.sum(), 1e-9))
        rows.append(d); maps[f"{item['patient_id']}_{s}"] = cam.astype(np.float16)
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.6)); ax[0].imshow(img, cmap="gray", vmin=-1, vmax=1); ax[0].set_title(f"{item['patient_id'][-14:]} slab {s}", fontsize=7)
        ax[1].imshow(img, cmap="gray", vmin=-1, vmax=1); ax[1].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        ax[1].set_title(f"Grad-CAM | P(malignant) {p:.2f} | true {d['label']}", fontsize=7); [q.axis("off") for q in ax]
        fig.tight_layout(); fig.savefig(out / f"{item['patient_id']}_{s}.png", dpi=110); plt.close(fig); shown.append((img, cam, d))
    S = pd.DataFrame(rows); S.to_csv(out / "gradcam_summary.csv", index=False); np.savez_compressed(out / "maps.npz", **maps)
    n = min(len(shown), 12); cols = 4; rws = int(np.ceil(n / cols))
    if n:
        fig, axs = plt.subplots(rws, cols, figsize=(3 * cols, 3 * rws), squeeze=False)
        for q in axs.ravel(): q.axis("off")
        for q, (img, cam, d) in zip(axs.ravel(), shown[:n]):
            q.imshow(img, cmap="gray", vmin=-1, vmax=1); q.imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1); q.set_title(f"{d['label'][:3]} P={d['score']:.2f}", fontsize=7)
        fig.tight_layout(); fig.savefig(out / "montage.png", dpi=110); plt.close(fig)
    if "cam_in_colon" in S:
        print(f"heat-map weight inside the colon outline: median {S.cam_in_colon.median():.2f} (colon covers {S.colon_frac.median():.2f} of the image)")
        print("  clearly above the colon share = looking at the colon; near or below it = looking elsewhere")
    print("wrote", out)


if __name__ == "__main__":
    main()
