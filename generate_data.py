import os
import numpy as np
import cv2

np.random.seed(2026)
KINDS = ['circle', 'square', 'triangle']


def make_one(kind, side=128):
    canvas = np.full((side, side), np.random.randint(5, 45), np.uint8)
    canvas = cv2.add(canvas, np.random.randint(0, 12, (side, side), dtype=np.uint8))
    lab = np.zeros((side, side), np.uint8)
    bright = np.random.randint(210, 256)

    # triangles reach farther from the center, so give them a bigger border
    pad = 45 if kind == 'triangle' else 35
    cx, cy = [int(v) for v in np.random.randint(pad, side - pad, 2)]

    if kind == 'circle':
        r = int(np.random.randint(10, 26))
        for t, c in ((lab, 1), (canvas, bright)):
            cv2.circle(t, (cx, cy), r, int(c), -1)
        return canvas, lab

    if kind == 'square':
        h = np.random.randint(12, 26)
        poly = np.int32(cv2.boxPoints(((cx, cy), (2.0 * h, 2.0 * h), np.random.uniform(0, 90))))
    else:
        sp = np.random.randint(30, 61)
        tri = np.array([[cx, cy - sp // 2], [cx - sp // 2, cy + sp // 2],
                        [cx + sp // 2, cy + sp // 2]], np.float32)
        rot = cv2.getRotationMatrix2D((float(cx), float(cy)), np.random.uniform(0, 360), 1.0)
        poly = cv2.transform(tri[None], rot)[0].astype(np.int32)

    for t, c in ((lab, 1), (canvas, bright)):
        cv2.drawContours(t, [poly], 0, int(c), -1)
    return canvas, lab


def build(n):
    pairs = [make_one(KINDS[i % 3]) for i in range(n)]
    xs = np.stack([p[0] for p in pairs])
    ys = np.stack([p[1] for p in pairs])
    return xs, ys, np.array([KINDS[i % 3] for i in range(n)])


if __name__ == '__main__':
    os.makedirs('dataset', exist_ok=True)
    for name, n in (('train', 1000), ('test', 200)):
        xs, ys, ks = build(n)
        for tag, arr in (('images', xs), ('masks', ys), ('shapes', ks)):
            np.save(f'dataset/{name}_{tag}.npy', arr)
        print(name, 'done:', xs.shape)