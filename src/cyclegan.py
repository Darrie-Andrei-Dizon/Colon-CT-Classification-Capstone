"""CycleGAN for the colon scans: train, then translate.  Built from the partner's notebook (same generator, discriminator,
learning rate, lambda_cycle 10, lambda_identity 5) with what the notebook was missing: the training loop, fixed seed,
checkpoint / resume, image pool, learning-rate decay, sample pictures, and a translate step.

    python cyclegan.py train     --npy-dir <prepared scans> --slices colon_slices.csv --split cohort_split.csv --out <run folder>
    python cyclegan.py translate --npy-dir <prepared scans> --slices colon_slices.csv --split cohort_split.csv \\
                                 --checkpoint <run folder>/last.pt --direction AtoB --out-dir <translated scans>

Domains:  A = StageII (cancer, contrast CT, collapsed colon)   B = CT Colonography (air-distended colon)
Training uses TRAIN-SIDE patients only (colon_dataset.UnpairedColon refuses val / test / challenge), colon slabs only,
one fixed intensity window (-400..400 HU), batch size 1, unpaired.
Training length is in iterations (one iteration = one A slab and one B slab).  The learning rate is constant for the
first half and falls linearly to zero in the second half, as in the original CycleGAN.  --max-hours stops cleanly and
saves, so a time-limited session (Kaggle: 12 h) never loses work; run the same command again to resume.

translate: applies one generator to every kept slab of the chosen source domain, for ALL patients of that source (the
generator never sees labels, so applying it to val / test is not leakage).  It writes a new folder with the same file names
and shapes as the prepared scans (float16 HU): translated kept slabs for the chosen domain, the other domain copied unchanged.
Feed that folder to colon_dataset (npy_dir) and to source_check_colon.py (--npy-dir, with --orig-dir = the originals).
Which direction to translate is a project decision (see advisor question 7); each domain is also one class, so LOOK at
translated tumours before trusting the result.

Colon-masked mode (v2):  add  --mask-dir <folder with the <series_uid>.npz colon masks>  [--dilate 4]  to train AND to translate.
The generators then only see and change the colon region (mask dilated by 4 pixels = 6 mm); everything outside is air in the
input and is forced back in the output, and at translation the new colon content is pasted into the original scan with a
feathered edge.  The body outline, body size and all tissue outside the colon region are therefore unchanged by construction.
This was added because the unconstrained version (v1) enlarged the body about 1.2 times on average and was rejected by
translate_qc.py in 68% of slabs.  Outputs of the two versions are not interchangeable: use a new --out folder for v2.

Needs: torch numpy pandas scipy matplotlib (and colon_dataset.py in the same folder)
"""
import argparse, itertools, shutil, sys, time
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colon_dataset as cd

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------- models (partner's design)
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
                                   nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch))

    def forward(self, x): return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, blocks=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(in_ch, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True),
            *[ResidualBlock(256) for _ in range(blocks)],
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(64, out_ch, 7), nn.Tanh())

    def forward(self, x): return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        def blk(i, o, norm=True): return [nn.Conv2d(i, o, 4, 2, 1)] + ([nn.InstanceNorm2d(o)] if norm else []) + [nn.LeakyReLU(0.2, True)]
        self.model = nn.Sequential(*blk(in_ch, 64, False), *blk(64, 128), *blk(128, 256),
                                   nn.Conv2d(256, 512, 4, 1, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(512, 1, 4, 1, 1))

    def forward(self, x): return self.model(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): nn.init.normal_(m.weight, 0.0, 0.02)


class ImagePool:
    """Keeps the last 50 generated images so the discriminator also sees older fakes (stabilises training)."""
    def __init__(self, size=50): self.size, self.buf = size, []

    def query(self, x):
        out = []
        for img in x:
            img = img.unsqueeze(0)
            if len(self.buf) < self.size: self.buf.append(img); out.append(img)
            elif np.random.rand() < 0.5:
                i = np.random.randint(self.size); out.append(self.buf[i].clone()); self.buf[i] = img
            else: out.append(img)
        return torch.cat(out)


# ----------------------------------------------------------------------------- colon-masked mode (--mask-dir)
@lru_cache(maxsize=16)
def _mask_vol(path):
    return np.load(path)["mask"]


def _dilate(m, n):
    return ndi.binary_dilation(m, iterations=n) if n else m


class MaskedUnpaired(cd.UnpairedColon):
    """Like UnpairedColon, but the generators only ever see (and can only change) the COLON REGION of each slab:
    the colon mask from select_colon.py, dilated by `dilate` pixels (1.5 mm each) to include the wall.
    Everything outside the region is air (-1) in the input and is forced back to air in the output, so the body outline,
    body size and everything outside the colon cannot be changed by construction.
    Item: dict(A, B, mA, mB): images (outside the region = -1) and the 0/1 region masks."""

    def __init__(self, *args, mask_dir, dilate=4, **kw):
        super().__init__(*args, **kw); self.mask_dir, self.dilate = Path(mask_dir), dilate

    def _get(self, ds, i):
        r = ds._r[i]; z = int(r["slab_idx"]); hu = np.asarray(cd._open(r["path"])[z], np.float32)
        m = _dilate(_mask_vol(str(self.mask_dir / f"{r['series_uid']}.npz"))[z], self.dilate)
        pair = torch.stack([torch.from_numpy(cd._pad_square(cd._window(hu), ds.size)),
                            torch.from_numpy(cd._pad_square(m.astype(np.float32) * 2 - 1, ds.size))])     # (2, H, W); mask coded -1 / +1
        if ds.augment: pair = cd._augment(pair)                       # image and mask are warped together
        mk = (pair[1:2] > 0).float(); x = torch.where(mk > 0, pair[0:1], torch.full_like(pair[0:1], -1.0))
        return (x.repeat(ds.channels, 1, 1), mk), r

    def __getitem__(self, i):
        (a, ma), _ = self._get(self.A, i % len(self.A)); (b, mb), _ = self._get(self.B, int(np.random.randint(len(self.B))))
        return dict(A=a, B=b, mA=ma, mB=mb)


def _comp(y, m):
    """Force everything outside the region back to air: y inside the mask, -1 outside."""
    return y if m is None else y * m - (1 - m)


def build(a, dev):
    nets = dict(G_AB=Generator(blocks=a.blocks), G_BA=Generator(blocks=a.blocks), D_A=Discriminator(), D_B=Discriminator())
    for n in nets.values(): n.apply(init_weights)
    return {k: v.to(dev) for k, v in nets.items()}


# ----------------------------------------------------------------------------- train
def save_samples(path, rA, fB, cA, rB, fA, cB):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ims = [(rA, "A real (StageII)"), (fB, "A to B"), (cA, "A back to A"), (rB, "B real (CT Colonography)"), (fA, "B to A"), (cB, "B back to B")]
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    for ax, (im, t) in zip(axs.ravel(), ims):
        ax.imshow(im[0, 0].detach().cpu().numpy(), cmap="gray", vmin=-1, vmax=1); ax.set_title(t, fontsize=8); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def train(a):
    cd.seed_everything(a.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); out = Path(a.out); (out / "samples").mkdir(parents=True, exist_ok=True)
    if a.mask_dir: ds = MaskedUnpaired(a.npy_dir, ("train",), a.slices, a.split, channels=1, size=a.size, augment=a.augment, mask_dir=a.mask_dir, dilate=a.dilate)
    else: ds = cd.UnpairedColon(a.npy_dir, ("train",), a.slices, a.split, channels=1, size=a.size, augment=a.augment)
    print(f"device {dev} | A (StageII) slabs {len(ds.A)} | B (CT Colonography) slabs {len(ds.B)} | {len(ds)} per epoch | train patients only" + (f" | colon-masked (dilate {a.dilate})" if a.mask_dir else ""), flush=True)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.workers, worker_init_fn=cd.seed_worker, drop_last=True,
                    pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    N = build(a, dev)
    oG = torch.optim.Adam(itertools.chain(N["G_AB"].parameters(), N["G_BA"].parameters()), lr=a.lr, betas=(0.5, 0.999))
    oDA = torch.optim.Adam(N["D_A"].parameters(), lr=a.lr, betas=(0.5, 0.999)); oDB = torch.optim.Adam(N["D_B"].parameters(), lr=a.lr, betas=(0.5, 0.999))
    decay = lambda it: 1.0 if it < a.iters / 2 else max(0.0, 2.0 * (1.0 - it / a.iters))
    sch = [torch.optim.lr_scheduler.LambdaLR(o, decay) for o in (oG, oDA, oDB)]
    it = 0; last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev)
        for k, n in N.items(): n.load_state_dict(ck[k])
        for o, k in ((oG, "oG"), (oDA, "oDA"), (oDB, "oDB")): o.load_state_dict(ck[k])
        for s, k in zip(sch, ("sG", "sDA", "sDB")): s.load_state_dict(ck[k])
        it = ck["iter"]; print(f"resumed from iteration {it}", flush=True)
        np.random.seed(a.seed + it); torch.manual_seed(a.seed + it)
    mse, l1 = nn.MSELoss(), nn.L1Loss(); poolA, poolB = ImagePool(), ImagePool(); log = []; t0 = time.time(); t_start = it
    logf = out / "train_log.csv"

    def save(final=False):
        ck = {k: n.state_dict() for k, n in N.items()}
        ck.update(oG=oG.state_dict(), oDA=oDA.state_dict(), oDB=oDB.state_dict(), sG=sch[0].state_dict(), sDA=sch[1].state_dict(),
                  sDB=sch[2].state_dict(), iter=it, args=vars(a))
        torch.save(ck, str(last) + ".tmp"); Path(str(last) + ".tmp").replace(last)
        gens = dict(G_AB=N["G_AB"].state_dict(), G_BA=N["G_BA"].state_dict(), iter=it, args=vars(a))
        if it % a.snapshot_every == 0 or final: torch.save(gens, out / f"generators_{it:06d}.pt")
        if log:
            pd.DataFrame(log).to_csv(logf, mode="a", header=not logf.exists(), index=False); log.clear()

    stop_at = time.time() + a.max_hours * 3600 if a.max_hours else None; done = False
    while it < a.iters and not done:
        for batch in dl:
            if it >= a.iters: break
            rA, rB = batch["A"].to(dev, non_blocking=True), batch["B"].to(dev, non_blocking=True)
            # --- generators
            oG.zero_grad(set_to_none=True)
            mA = batch["mA"].to(dev, non_blocking=True) if a.mask_dir else None; mB = batch["mB"].to(dev, non_blocking=True) if a.mask_dir else None
            idA, idB = _comp(N["G_BA"](rA), mA), _comp(N["G_AB"](rB), mB); l_id = (l1(idA, rA) + l1(idB, rB)) * a.lambda_identity
            fB, fA = _comp(N["G_AB"](rA), mA), _comp(N["G_BA"](rB), mB)
            pB, pA = N["D_B"](fB), N["D_A"](fA); l_gan = mse(pB, torch.ones_like(pB)) + mse(pA, torch.ones_like(pA))
            cA, cB = _comp(N["G_BA"](fB), mA), _comp(N["G_AB"](fA), mB); l_cyc = (l1(cA, rA) + l1(cB, rB)) * a.lambda_cycle
            lG = l_gan + l_cyc + l_id; lG.backward(); oG.step()
            # --- discriminators (real vs pooled fakes)
            oDA.zero_grad(set_to_none=True); pr, pf = N["D_A"](rA), N["D_A"](poolA.query(fA.detach()))
            lDA = 0.5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf))); lDA.backward(); oDA.step()
            oDB.zero_grad(set_to_none=True); pr, pf = N["D_B"](rB), N["D_B"](poolB.query(fB.detach()))
            lDB = 0.5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf))); lDB.backward(); oDB.step()
            for s in sch: s.step()
            it += 1
            if it % a.log_every == 0:
                rate = (it - t_start) / (time.time() - t0); eta = (a.iters - it) / rate / 3600
                log.append(dict(iter=it, G=lG.item(), gan=l_gan.item(), cycle=l_cyc.item(), identity=l_id.item(), D_A=lDA.item(), D_B=lDB.item(), lr=sch[0].get_last_lr()[0]))
                print(f"iter {it}/{a.iters} | G {lG.item():.3f} cyc {l_cyc.item():.3f} D_A {lDA.item():.3f} D_B {lDB.item():.3f} | {rate:.2f} it/s | about {eta:.1f} h left", flush=True)
            if it % a.sample_every == 0: save_samples(out / "samples" / f"iter_{it:06d}.png", rA, fB, cA, rB, fA, cB)
            if it % a.save_every == 0: save()
            if stop_at and time.time() > stop_at:
                print(f"--max-hours reached at iteration {it}; saving. Run the same command again to continue.", flush=True); done = True; break
    save(final=it >= a.iters); print("done" if it >= a.iters else "stopped early (resumable)", flush=True)


# ----------------------------------------------------------------------------- translate
@torch.no_grad()
def translate(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); ck = torch.load(a.checkpoint, map_location=dev)
    size = ck["args"]["size"]; G = Generator(blocks=ck["args"]["blocks"]).to(dev); G.load_state_dict(ck["G_AB" if a.direction == "AtoB" else "G_BA"]); G.eval()
    src = "StageII" if a.direction == "AtoB" else "CTC"
    masked, dil = bool(ck["args"].get("mask_dir")), ck["args"].get("dilate", 0)
    if masked and not a.mask_dir: raise SystemExit("this checkpoint was trained colon-masked: pass --mask-dir (the folder with <series_uid>.npz masks)")
    sl = pd.read_csv(a.slices); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); n = sl.series_uid.nunique(); done = 0
    print(f"translating {src} scans with {a.direction} (checkpoint iteration {ck['iter']}) on {dev}; {n} series", flush=True)
    for k, (uid, g) in enumerate(sl.groupby("series_uid"), 1):
        f_out = out / f"{uid}.npy"
        if f_out.exists(): continue
        orig = np.load(Path(a.npy_dir) / f"{uid}.npy")
        if g.source.iloc[0] != src: shutil.copy(Path(a.npy_dir) / f"{uid}.npy", f_out); continue
        vol = orig.astype(np.float32); h, w = vol.shape[1:]
        if h > size or w > size: raise ValueError(f"scan {uid[-10:]} is {h}x{w}, larger than the training size {size}")
        oy, ox = (size - h) // 2, (size - w) // 2; z = np.sort(g.slab_idx.values)
        if masked: mvol = _mask_vol(str(Path(a.mask_dir) / f"{uid}.npz"))
        for i in range(0, len(z), a.batch):
            zz = z[i:i + a.batch]
            if masked:
                dm = [_dilate(mvol[j], dil) for j in zz]
                x = torch.from_numpy(np.stack([cd._pad_square(np.where(m, cd._window(vol[j]), -1.0).astype(np.float32), size) for j, m in zip(zz, dm)])).unsqueeze(1).to(dev)
            else:
                x = torch.from_numpy(np.stack([cd._pad_square(cd._window(vol[j]), size) for j in zz])).unsqueeze(1).to(dev)
            y = G(x)[:, 0, oy:oy + h, ox:ox + w].cpu().numpy()
            for k2, (j, yy) in enumerate(zip(zz, y)):
                if masked:                                         # feathered paste: new colon content inside, original HU everywhere else
                    soft = ndi.gaussian_filter(dm[k2].astype(np.float32), 1.5); vol[j] = soft * cd.to_hu(yy) + (1 - soft) * vol[j]
                else: vol[j] = cd.to_hu(yy)
        np.save(f_out, vol.astype(np.float16)); done += 1
        if k % 10 == 0 or k == n: print(f"{k}/{n}", flush=True)
    print(f"wrote {out} ({done} series translated, others copied unchanged)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--npy-dir", required=True, help="prepared scans (<series_uid>.npy)")
        p.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
        p.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    t = sub.add_parser("train"); common(t); t.add_argument("--out", required=True, help="run folder (checkpoints, log, sample pictures)")
    t.add_argument("--iters", type=int, default=60000, help="total iterations, batch size 1")
    t.add_argument("--lr", type=float, default=2e-4); t.add_argument("--lambda-cycle", type=float, default=10.0)
    t.add_argument("--lambda-identity", type=float, default=5.0); t.add_argument("--blocks", type=int, default=6)
    t.add_argument("--size", type=int, default=256); t.add_argument("--augment", action="store_true")
    t.add_argument("--seed", type=int, default=cd.SEED); t.add_argument("--workers", type=int, default=2)
    t.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this many hours (0 = no limit)")
    t.add_argument("--log-every", type=int, default=100); t.add_argument("--sample-every", type=int, default=1000)
    t.add_argument("--mask-dir", default=None, help="colon masks (<series_uid>.npz from select_colon.py): train and translate on the colon region only")
    t.add_argument("--dilate", type=int, default=4, help="grow the colon region by this many pixels (1.5 mm each) to include the wall")
    t.add_argument("--save-every", type=int, default=2000); t.add_argument("--snapshot-every", type=int, default=10000)
    r = sub.add_parser("translate"); common(r); r.add_argument("--checkpoint", required=True); r.add_argument("--out-dir", required=True)
    r.add_argument("--direction", choices=["AtoB", "BtoA"], required=True, help="AtoB: StageII looks like CT Colonography; BtoA: the reverse")
    r.add_argument("--batch", type=int, default=16); r.add_argument("--mask-dir", default=None, help="needed if the checkpoint was trained with --mask-dir")
    a = ap.parse_args(); train(a) if a.cmd == "train" else translate(a)
