import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # save figures to files, no display needed
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(2026)
rs = np.random.RandomState(2026)
KINDS = ['circle', 'square', 'triangle']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ShapeSet(Dataset):
    def __init__(self, part):
        self.xs = np.load(f'dataset/{part}_images.npy')
        self.ys = np.load(f'dataset/{part}_masks.npy')
        self.ks = np.load(f'dataset/{part}_shapes.npy')

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x = torch.from_numpy(self.xs[i]).float().unsqueeze(0) / 255.0
        y = torch.from_numpy(self.ys[i]).float().unsqueeze(0)
        return x, y, str(self.ks[i])


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(True),
                         nn.Conv2d(b, b, 3, padding=1), nn.ReLU(True))


class MiniUNet(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.e1, self.e2 = block(1, w), block(w, 2 * w)
        self.mp = nn.MaxPool2d(2)
        self.neck = block(2 * w, 4 * w)
        self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.d2 = block(4 * w, 2 * w)
        self.u1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.d1 = block(2 * w, w)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.mp(a))
        c = self.neck(self.mp(b))
        c = self.d2(torch.cat([self.u2(c), b], 1))
        c = self.d1(torch.cat([self.u1(c), a], 1))
        return torch.sigmoid(self.head(c))


crit = nn.BCELoss()


def evaluate(net, dl):
    """returns mean loss and [tp, fp, fn, tn] counts for each shape type"""
    net.eval()
    cnt = {k: np.zeros(4) for k in KINDS}
    tot = 0.0
    with torch.no_grad():
        for x, y, ks in dl:
            x, y = x.to(dev), y.to(dev)
            p = net(x)
            tot += crit(p, y).item() * len(x)
            pb, gt = p > 0.5, y > 0.5
            for k in KINDS:
                sel = torch.tensor([s == k for s in ks], device=dev)
                u, v = pb[sel], gt[sel]
                cnt[k] += [(u & v).sum().item(), (u & ~v).sum().item(),
                           (~u & v).sum().item(), (~u & ~v).sum().item()]
    net.train()
    return tot / len(dl.dataset), cnt


def report(cnt):
    tp, fp, fn, tn = sum(cnt.values())
    return (tp + tn) / (tp + fp + fn + tn), tp / (tp + fp + fn + 1e-8), (tp, fp, fn, tn)


def main():
    tr, te = ShapeSet('train'), ShapeSet('test')
    dl_tr = DataLoader(tr, batch_size=16, shuffle=True)
    dl_te = DataLoader(te, batch_size=16)

    net = MiniUNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    step_loss, te_curve, EPOCHS = [], [], 15

    for ep in range(EPOCHS):
        run = 0.0
        for x, y, _ in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            l = crit(net(x), y)
            l.backward()
            opt.step()
            run += l.item() * len(x)
            step_loss.append(l.item())
        tl, _ = evaluate(net, dl_te)
        te_curve.append(tl)
        print(f'ep {ep + 1:02d} | train {run / len(tr):.4f} | test {tl:.4f}')

    # metrics on both sets (to check over/underfitting)
    for name, dl in (('Train', DataLoader(tr, batch_size=16)), ('Test', dl_te)):
        _, cnt = evaluate(net, dl)
        acc, iou, (tp, fp, fn, tn) = report(cnt)
        print(f'\n[{name}] TP={tp:.0f} FP={fp:.0f} FN={fn:.0f} TN={tn:.0f}')
        print(f'[{name}] Acc={acc * 100:.2f}%  IoU={iou:.4f}')
        if name == 'Test':
            for k in KINDS:
                a, i, _ = report({k: cnt[k]})
                print(f'   {k:8s} Acc={a * 100:.2f}%  IoU={i:.4f}')

    # learning curve
    per_ep = len(dl_tr)
    plt.figure(figsize=(7, 3.5))
    plt.plot(step_loss, lw=1, label='train (per update)')
    plt.plot([per_ep * (i + 1) for i in range(EPOCHS)], te_curve, 'o-', ms=3, label='test (per epoch)')
    plt.xlabel('Training updates'); plt.ylabel('BCE loss')
    plt.grid(ls='--', alpha=.5); plt.legend(); plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=200)

    # 2 random test samples per shape
    ids = []
    for k in KINDS:
        pool = np.where(te.ks == k)[0]
        ids += list(rs.choice(pool, 2, replace=False))
    fig, ax = plt.subplots(6, 3, figsize=(7, 13))
    net.eval()
    with torch.no_grad():
        for r, i in enumerate(ids):
            x, y, k = te[i]
            p = (net(x[None].to(dev)) > 0.5).float().cpu().squeeze().numpy()
            for c, (im, t) in enumerate(((x[0], f'Input ({k})'), (y[0], 'Ground truth'), (p, 'Prediction'))):
                ax[r, c].imshow(im, cmap='gray'); ax[r, c].set_title(t, fontsize=9); ax[r, c].axis('off')
    plt.tight_layout()
    plt.savefig('samples.png', dpi=200)


if __name__ == '__main__':
    main()