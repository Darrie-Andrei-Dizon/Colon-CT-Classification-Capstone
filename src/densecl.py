"""DenseCL self-supervised pretraining (Wang et al., 2021) for DenseNet-121 and ResNet-50 on the colon slices.

    python densecl.py --arch densenet121 --npy-dir <scans> --slices colon_slices.csv --split cohort_split.csv --out <run folder>

Used for experiments D (npy-dir = original scans) and E (npy-dir = CycleGAN-translated scans) of Table 1.
Run once per backbone and per scan folder: 4 runs in total.  Resume by running the same command again.

What it does (manuscript 3.4.3.1 and 3.5, stage 3)
  * starts from ImageNet weights and updates the WHOLE backbone (nothing frozen except the batch-norm statistics, see backbones.py)
  * no labels are used; only TRAIN-SIDE patients (colon_dataset.load_rows is called with the train-side splits only)
  * two augmented views of one slice (random crop, left-right flip, intensity jitter) go through a query encoder and a
    momentum key encoder; the loss is  (1 - lam) * global InfoNCE  +  lam * dense InfoNCE
    dense part: every location of the query feature map is matched to its most similar location in the other view,
    and those pairs are pulled together while other images' pooled features (queue) are pushed away
  * records what the manuscript says must be reported: duration, batch size, temperature, which layers are updated
    (run_config.json) and the loss per epoch (train_log.csv)

Outputs: <out>/encoder.pt   (query-encoder weights, load with backbones.load_pretrained)
         <out>/last.pt      (everything needed to resume)
         <out>/train_log.csv, <out>/run_config.json

Flip: ON here (manuscript 3.5 lists it for DenseCL).  It is harmless for a label-free local-feature objective and gives
free diversity.  The classifier does NOT flip (see classify.py): there it would only teach the network that left and
right colon are interchangeable.
Time: --max-hours stops cleanly after a finished epoch and saves.  Measure the first epoch before choosing --epochs.
Needs: torch torchvision numpy pandas (and colon_dataset.py, backbones.py in the same folder)
"""
import argparse, copy, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colon_dataset as cd
from backbones import Encoder, ARCHS

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------- data
class PretrainSlices(Dataset):
    """Unlabelled train-side colon slabs, two augmented views each."""

    def __init__(self, npy_dir, slices_csv=None, split_csv=None, size=256, crop=224, scale=(0.3, 1.0), flip=True, jitter=True):
        self.rows = cd.load_rows(slices_csv or ROOT / "data/processed/colon_slices.csv",
                                 split_csv or ROOT / "data/processed/cohort_split.csv", npy_dir, sorted(cd.TRAIN_SIDE))
        if not len(self.rows): raise ValueError("no train-side slabs found")
        self._r = self.rows[["path", "slab_idx"]].to_dict("records")
        self.size, self.crop, self.scale, self.flip, self.jitter = size, crop, scale, flip, jitter

    def __len__(self): return len(self._r)

    def _view(self, x):                                   # x: (1, H, W) in [-1, 1]
        H, W = x.shape[-2:]
        for _ in range(10):
            area = H * W * np.random.uniform(*self.scale); ratio = math.exp(np.random.uniform(math.log(3 / 4), math.log(4 / 3)))
            w, h = int(round(math.sqrt(area * ratio))), int(round(math.sqrt(area / ratio)))
            if 0 < w <= W and 0 < h <= H: break
        else:
            w, h = W, H
        i, j = np.random.randint(0, H - h + 1), np.random.randint(0, W - w + 1)
        x = F.interpolate(x[None, :, i:i + h, j:j + w], size=(self.crop, self.crop), mode="bilinear", align_corners=False)[0]
        if self.flip and np.random.rand() < 0.5: x = x.flip(-1)
        if self.jitter and np.random.rand() < 0.8:        # brightness / contrast jitter, on the [-1, 1] scale
            x = (x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1)).clamp(-1, 1)
        return x.repeat(3, 1, 1)

    def __getitem__(self, i):
        r = self._r[i]; hu = np.asarray(cd._open(r["path"])[int(r["slab_idx"])], np.float32)
        x = torch.from_numpy(cd._pad_square(cd._window(hu), self.size)).unsqueeze(0)
        return self._view(x), self._view(x)


# ----------------------------------------------------------------------------- model
class Heads(nn.Module):
    """Global head (pooled feature -> dim) and dense head (1x1 convolutions, one vector per location)."""

    def __init__(self, ch, hidden=2048, dim=128):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(ch, hidden), nn.ReLU(True), nn.Linear(hidden, dim))
        self.d = nn.Sequential(nn.Conv2d(ch, hidden, 1), nn.ReLU(True), nn.Conv2d(hidden, dim, 1))

    def forward(self, f):
        g = F.normalize(self.g(f.mean((2, 3))).float(), dim=1)
        d = F.normalize(self.d(f).float().flatten(2), dim=1)          # (B, dim, S)
        return g, d


class DenseCL(nn.Module):
    def __init__(self, arch, weights, dim=128, K=16384, m=0.99, T=0.2, lam=0.5, hidden=2048, freeze_bn=True):
        super().__init__()
        self.K, self.m, self.T, self.lam = K, m, T, lam
        self.enc_q = Encoder(arch, weights, freeze_bn); self.head_q = Heads(self.enc_q.out_ch, hidden, dim)
        self.enc_k, self.head_k = copy.deepcopy(self.enc_q), copy.deepcopy(self.head_q)
        for p in list(self.enc_k.parameters()) + list(self.head_k.parameters()): p.requires_grad = False
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue2", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum(self):
        for a, b in ((self.enc_q, self.enc_k), (self.head_q, self.head_k)):
            for pq, pk in zip(a.parameters(), b.parameters()): pk.mul_(self.m).add_(pq.detach(), alpha=1 - self.m)

    @torch.no_grad()
    def _enqueue(self, kg, kd):
        if not (torch.isfinite(kg).all() and torch.isfinite(kd).all()): return        # never put NaN keys into the queues
        n, p = kg.shape[0], int(self.ptr)
        assert self.K % n == 0, "queue size must be a multiple of the batch size"
        self.queue[:, p:p + n] = kg.T; self.queue2[:, p:p + n] = kd.T; self.ptr[0] = (p + n) % self.K

    def forward(self, im_q, im_k):
        fq = self.enc_q(im_q); qg, qd = self.head_q(fq)
        with torch.no_grad():
            self._momentum(); fk = self.enc_k(im_k); kg, kd = self.head_k(fk)
        # global InfoNCE
        logits = torch.cat([(qg * kg).sum(1, keepdim=True), qg @ self.queue.clone().detach()], 1) / self.T
        tgt = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss_g = F.cross_entropy(logits, tgt); acc = (logits.argmax(1) == 0).float().mean()
        # dense InfoNCE: positive = key location most similar (backbone features) to the query location
        with torch.no_grad():
            sim = torch.bmm(F.normalize(fq.float().flatten(2), dim=1).transpose(1, 2), F.normalize(fk.float().flatten(2), dim=1))
            idx = sim.argmax(2)                                                   # (B, S)
            kd_match = torch.gather(kd, 2, idx.unsqueeze(1).expand(-1, kd.shape[1], -1))
        l_pos = (qd * kd_match).sum(1).reshape(-1, 1)
        qd_flat = qd.permute(0, 2, 1).reshape(-1, qd.shape[1])
        logits_d = torch.cat([l_pos, qd_flat @ self.queue2.clone().detach()], 1) / self.T
        loss_d = F.cross_entropy(logits_d, torch.zeros(logits_d.shape[0], dtype=torch.long, device=logits_d.device))
        with torch.no_grad(): self._enqueue(kg, F.normalize(kd.mean(2), dim=1))
        return (1 - self.lam) * loss_g + self.lam * loss_d, loss_g.detach(), loss_d.detach(), acc


# ----------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--weights", default="imagenet", help="imagenet | none | path to a torchvision state dict")
    ap.add_argument("--npy-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = 0.03 * batch / 256 (SGD, momentum 0.9, cosine)")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.2); ap.add_argument("--momentum", type=float, default=0.99)
    ap.add_argument("--queue", type=int, default=16384); ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--no-flip", action="store_true"); ap.add_argument("--train-bn", action="store_true")
    ap.add_argument("--no-amp", action="store_true"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=cd.SEED); ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--log-every", type=int, default=50); ap.add_argument("--max-iters", type=int, default=0, help="smoke test only")
    a = ap.parse_args()

    cd.seed_everything(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda" and not a.no_amp
    ds = PretrainSlices(a.npy_dir, a.slices, a.split, a.size, a.crop, flip=not a.no_flip)
    dl = DataLoader(ds, a.batch, shuffle=True, drop_last=True, num_workers=a.workers, worker_init_fn=cd.seed_worker,
                    persistent_workers=a.workers > 0)
    ipe = len(dl); lr0 = a.lr or 0.03 * a.batch / 256
    model = DenseCL(a.arch, a.weights, K=a.queue, m=a.momentum, T=a.temperature, lam=a.lam, freeze_bn=not a.train_bn).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr0, momentum=0.9, weight_decay=a.wd); scaler = torch.amp.GradScaler(enabled=amp)
    log_path, start = out / "train_log.csv", 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); start = ck["epoch"]; print(f"resumed after epoch {start}")
    elif log_path.exists(): log_path.unlink()
    print(f"device {dev.type} | amp {amp} | {a.arch} | train-side slabs {len(ds)} | {ipe} iterations per epoch | lr {lr0:.5f}")
    t_start, t_used = time.time(), 0.0
    if (out / "run_config.json").exists(): t_used = json.load(open(out / "run_config.json")).get("hours_so_far", 0) * 3600
    ep_times = []; bad = 0
    for ep in range(start, a.epochs):
        for g in opt.param_groups: g["lr"] = lr0 * 0.5 * (1 + math.cos(math.pi * ep / a.epochs))
        model.train(); t0 = time.time(); tot = np.zeros(4); n = 0
        for it, (vq, vk) in enumerate(dl, 1):
            vq, vk = vq.to(dev, non_blocking=True), vk.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss, lg, ld, acc = model(vq, vk)
            if not torch.isfinite(loss):                  # skip a bad batch; stop if it keeps happening
                bad += 1; opt.zero_grad(set_to_none=True)
                if bad == 1: print(f"non-finite loss at epoch {ep + 1} iteration {it}", flush=True)
                if bad > 20: raise SystemExit("more than 20 non-finite losses: stopping. Run again with --no-amp (full precision).")
                continue
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0); scaler.step(opt); scaler.update()
            tot += np.array([loss.item(), lg.item(), ld.item(), acc.item()]); n += 1
            if it % a.log_every == 0:
                print(f"epoch {ep + 1}/{a.epochs} it {it}/{ipe} | loss {tot[0] / n:.3f} global {tot[1] / n:.3f} dense {tot[2] / n:.3f} "
                      f"| {it * a.batch / (time.time() - t0):.1f} img/s", flush=True)
            if a.max_iters and it >= a.max_iters: break
        tot /= max(n, 1); ep_times.append(time.time() - t0)
        with open(log_path, "a") as f:
            if ep == 0 or not log_path.stat().st_size: f.write("epoch,loss,loss_global,loss_dense,top1_global,seconds\n")
            f.write(f"{ep + 1},{tot[0]:.4f},{tot[1]:.4f},{tot[2]:.4f},{tot[3]:.4f},{ep_times[-1]:.0f}\n")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=ep + 1), out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")
        hours = (t_used + time.time() - t_start) / 3600; done = ep + 1 == a.epochs
        cfg = dict(vars(a), hours_so_far=round(hours, 3), epochs_done=ep + 1, iterations_per_epoch=ipe, learning_rate_start=lr0,
                   layers_updated="all backbone layers (batch-norm statistics frozen)" if not a.train_bn else "all backbone layers",
                   views="random resized crop (scale 0.3-1), left-right flip" + (" off" if a.no_flip else " on") + ", brightness/contrast jitter")
        json.dump(cfg, open(out / "run_config.json", "w"), indent=1)
        print(f"epoch {ep + 1} done | loss {tot[0]:.3f} | {ep_times[-1] / 60:.1f} min | {hours:.2f} h used", flush=True)
        if a.max_hours and not done and hours + 1.05 * np.mean(ep_times) / 3600 > a.max_hours:
            print(f"stopping before the time limit; run the same command again to continue (epoch {ep + 1} of {a.epochs} saved)"); break
        if a.max_iters: break
    finished = json.load(open(out / "run_config.json"))["epochs_done"] >= a.epochs
    if finished or a.max_iters:
        torch.save(model.enc_q.state_dict(), out / "encoder.pt"); print("saved", out / "encoder.pt")

if __name__ == "__main__":
    main()
