"""Model-free camera measurements from the depth buffer: height above ground, pitch, distance to our kart."""
import sys, numpy as np
sys.path.insert(0, ".")
import lak.camera as C

def measure(z, cam, H, W):
    u = (np.arange(W) + 0.5) / W * 2 - 1; v = 1 - (np.arange(H) + 0.5) / H * 2
    X = np.stack([u[None, :] / cam.w * z, v[:, None] / cam.h * z, z], -1)
    m = np.zeros((H, W), bool); m[int(H * 0.62):, :] = True; m[:, int(W * 0.3):int(W * 0.7)] = False
    m &= z < 6
    P = X[m]
    if len(P) < 500: return np.nan, np.nan, np.nan
    c = P.mean(0); n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    for _ in range(2):
        keep = np.abs((P - c) @ n) < 0.05
        if keep.sum() < 300: break
        c = P[keep].mean(0); n = np.linalg.svd(P[keep] - c, full_matrices=False)[2][-1]
    if n[1] < 0: n = -n
    ego = z[int(H * 0.45):int(H * 0.8), W // 2 - 3:W // 2 + 4]
    return abs(c @ n), np.degrees(np.arcsin(-n[2])), np.percentile(ego, 8)

if __name__ == "__main__":
    d = {k: v for k, v in np.load(sys.argv[1]).items()}
    H, W = d["depth"].shape[1:]
    cam = C.ChaseCamera(W, H)
    lo, hi, st = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    for i in range(lo, hi, st):
        h, p, r = measure(cam.linear_depth(d["depth"][i]), cam, H, W)
        print(f"{i:4d} speed {d['speed'][i]:5.1f}  height {h:.2f}  pitch {p:5.1f}  ego_rear {r:.2f}")
