"""E2: per-class IoU and depth error of a perception variant against GT, on saved frames."""
import sys, time, json, numpy as np
sys.path.insert(0, ".")
from PIL import Image
from lak.scene import CLASS_NAMES, PALETTE
from lak.perception_real import RealPerception
track = sys.argv[1] if len(sys.argv) > 1 else "lighthouse"
d = np.load(f"data/frames_{track}.npz")
rgb, gt, gd, ego = d["rgb"], d["cls"], d["depth"].astype(np.float32), d["ego"]
p = RealPerception()
preds, invs, t0 = [], [], time.time()
for im in rgb:
    c, inv = p.raw(im); preds.append(c); invs.append(inv)
ms = (time.time() - t0) / len(rgb) * 1e3
preds, invs = np.array(preds), np.array(invs)
valid = ~ego
iou = {}
for k, name in enumerate(CLASS_NAMES):
    inter = ((preds == k) & (gt == k) & valid).sum(); union = (((preds == k) | (gt == k)) & valid).sum()
    iou[name] = round(float(inter / max(union, 1)), 3)
# depth: fit 1/z = a*inv + b on the first half of frames (calibration set), test on the second half
h = len(rgb) // 2
m = valid & (gd < 80)
A = np.stack([invs[:h][m[:h]], np.ones(m[:h].sum())], 1)
a, b = np.linalg.lstsq(A, 1.0 / gd[:h][m[:h]], rcond=None)[0]
pred_z = np.clip(1.0 / np.maximum(a * invs[h:] + b, 1e-3), 0, 120)   # sky / far pixels: cap like the eval range
absrel = float(np.mean(np.abs(pred_z[m[h:]] - gd[h:][m[h:]]) / gd[h:][m[h:]]))
res = dict(track=track, frames=len(rgb), iou=iou, miou=round(float(np.mean(list(iou.values()))), 3),
           depth_absrel=round(absrel, 3), depth_scale=[float(a), float(b)], ms_per_frame=round(ms, 1), device=p.dev)
print(json.dumps(res)); json.dump(res, open(f"out/e2_real_{track}.json", "w"))
tiles = []
for i in np.linspace(0, len(rgb) - 1, 4).astype(int):
    small = np.asarray(Image.fromarray(rgb[i]).resize((320, 240)))
    z = 1.0 / np.maximum(a * invs[i] + b, 1e-3)
    dv = lambda x: np.stack([(255 * np.clip(1 - np.log1p(x) / np.log1p(120), 0, 1)).astype(np.uint8)] * 3, -1)
    tiles.append(np.concatenate([small, PALETTE[gt[i]], PALETTE[preds[i]], dv(gd[i]), dv(z)], 1))
Image.fromarray(np.concatenate(tiles, 0)).save(f"out/e2_real_{track}.png")
