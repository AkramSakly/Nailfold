"""
train_unet.py  --  PARTIE B : SEGMENTATION (dataset Segmentation)
========================================================================
Entraine un U-Net pour segmenter les capillaires dans les images nailfold,
a partir des 70 paires image/masque du dossier 'Segmentation'.

Architecture : U-Net classique (encodeur-decodeur avec skip connections),
exactement le type de reseau utilise dans les articles de segmentation
cites par la reference (Hariyani et al., Liu et al., Chen et al.).

Metriques : IoU (Intersection over Union) et Dice - les memes que dans
ces articles, ce qui permet une comparaison directe dans le rapport.

Perte : combinaison BCE + Dice loss (adaptee au fort desequilibre :
les capillaires ne representent qu'environ 1% des pixels).

UTILISATION :
  python segmentation/train_unet.py --dossier Segmentation --epochs 40

Le dossier doit contenir seg_images/ et seg_masks/.
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMG = 256   # taille de travail (les images font 1280x1024)


# ---------------------------------------------------------------- DATASET
def find_dir(root, keyword):
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if keyword in d.lower():
                return os.path.join(dirpath, d)
    return None


def list_pairs(dossier):
    img_dir = find_dir(dossier, "image")
    mask_dir = find_dir(dossier, "mask")
    if img_dir is None or mask_dir is None:
        raise RuntimeError("seg_images/ ou seg_masks/ introuvable dans "
                           + dossier)
    pairs = []
    for img_path in sorted(glob.glob(os.path.join(img_dir, "*"))):
        base = os.path.splitext(os.path.basename(img_path))[0]
        # masque = <base>mask.jpg
        cand = os.path.join(mask_dir, base + "mask.jpg")
        if not os.path.exists(cand):
            # essai sans extension fixe
            hits = glob.glob(os.path.join(mask_dir, base + "mask.*"))
            cand = hits[0] if hits else None
        if cand and os.path.exists(cand):
            pairs.append((img_path, cand))
    return pairs


class SegDS(Dataset):
    def __init__(self, pairs, train):
        self.pairs, self.train = pairs, train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("RGB").resize((IMG, IMG))
        mask = Image.open(mp).convert("L").resize((IMG, IMG))
        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy(); mask = mask[:, ::-1].copy()
        if self.train and np.random.rand() < 0.5:
            img = img[::-1, :, :].copy(); mask = mask[::-1, :].copy()
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


# ---------------------------------------------------------------- U-NET
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
    def forward(self, x): return self.c(x)


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.d1 = DoubleConv(3, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.u4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.u3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.u1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        c1 = self.d1(x); c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2)); c4 = self.d4(self.pool(c3))
        b = self.bott(self.pool(c4))
        x = self.u4(torch.cat([self.up4(b), c4], 1))
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.out(x)


# ---------------------------------------------------------------- PERTE + METRIQUES
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((2, 3)) + eps
    den = p.sum((2, 3)) + target.sum((2, 3)) + eps
    return (1 - num / den).mean()


def combo_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) \
        + dice_loss(logits, target)


@torch.no_grad()
def iou_dice(logits, target, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum((1, 2, 3))
    union = ((p + target) > 0).float().sum((1, 2, 3))
    iou = ((inter + eps) / (union + eps)).mean().item()
    dice = ((2 * inter + eps) /
            (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean().item()
    return iou, dice


# ---------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", default="Segmentation")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs_seg")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil : {device}")

    pairs = list_pairs(args.dossier)
    print(f"Paires image/masque trouvees : {len(pairs)}")
    if len(pairs) == 0:
        raise RuntimeError("Aucune paire trouvee.")

    idx = np.arange(len(pairs)); np.random.shuffle(idx)
    n_val = max(1, int(len(pairs) * args.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr_pairs = [pairs[i] for i in tr_idx]
    va_pairs = [pairs[i] for i in val_idx]
    print(f"Entrainement : {len(tr_pairs)}  |  Validation : {len(va_pairs)}")

    tr = DataLoader(SegDS(tr_pairs, True), batch_size=args.batch_size,
                    shuffle=True, num_workers=0)
    va = DataLoader(SegDS(va_pairs, False), batch_size=args.batch_size,
                    num_workers=0)

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dice = 0.0
    os.makedirs(args.outdir, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0
        for x, m in tr:
            x, m = x.to(device), m.to(device)
            opt.zero_grad(); loss = combo_loss(model(x), m)
            loss.backward(); opt.step(); tl += loss.item() * x.size(0)
        tl /= len(tr.dataset)
        # validation
        model.eval(); ious, dices = [], []
        with torch.no_grad():
            for x, m in va:
                x, m = x.to(device), m.to(device)
                i, d = iou_dice(model(x), m)
                ious.append(i); dices.append(d)
        miou, mdice = float(np.mean(ious)), float(np.mean(dices))
        if ep % 5 == 0 or ep == 1:
            print(f"  epoque {ep:3d}  loss={tl:.4f}  IoU={miou:.4f}  Dice={mdice:.4f}")
        if mdice > best_dice:
            best_dice = mdice
            torch.save(model.state_dict(),
                       os.path.join(args.outdir, "unet_best.pt"))

    print(f"\nMeilleur Dice (validation) : {best_dice:.4f}")
    print(f"Modele sauvegarde : {args.outdir}/unet_best.pt")

    # ---- sauvegarder quelques predictions en images ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        model.load_state_dict(torch.load(
            os.path.join(args.outdir, "unet_best.pt"), map_location=device))
        model.eval()
        x, m = next(iter(va))
        with torch.no_grad():
            pred = (torch.sigmoid(model(x.to(device))) > 0.5).float().cpu()
        n = min(3, x.size(0))
        fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1: ax = ax[None, :]
        for k in range(n):
            ax[k, 0].imshow(x[k].permute(1, 2, 0).numpy()); ax[k, 0].set_title("Image")
            ax[k, 1].imshow(m[k, 0].numpy(), cmap="gray"); ax[k, 1].set_title("Masque reel")
            ax[k, 2].imshow(pred[k, 0].numpy(), cmap="gray"); ax[k, 2].set_title("Prediction")
            for a in ax[k]: a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "exemples_segmentation.png"), dpi=120)
        print(f"Exemples visuels : {args.outdir}/exemples_segmentation.png")
    except Exception as e:
        print("(graphique non genere :", e, ")")


if __name__ == "__main__":
    main()