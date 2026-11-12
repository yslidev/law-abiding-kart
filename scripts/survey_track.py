"""Survey a track's road surface: let the built-in AI drive one lap and fit a plane per driveline quad."""
import sys, numpy as np
sys.path.insert(0, ".")
from lak.env import KartEnv
from lak.perception_gt import RoadMap, survey_path
from lak.camera import quat_to_mat

track = sys.argv[1] if len(sys.argv) > 1 else "lighthouse"
env = KartEnv(track, n_karts=1, ego_ai=True, laps=2)
obs = env.reset()
rm = RoadMap(env.nodes, env.widths)
samples = [[] for _ in range(len(env.nodes))]
prev_speed, used = 0.0, 0
for step in range(6000):
    obs = env.step(None)
    steady = obs.speed > 0.6 * env.max_speed and abs(obs.speed - prev_speed) < 0.4   # camera model is exact in steady driving
    prev_speed = obs.speed
    if obs.kart.finished_laps >= 2 or obs.kart.has_finished_race:
        break
    if step % 2 or not steady:
        continue
    used += 1
    cam = obs.cam
    z = cam.linear_depth(obs.depth_buf)
    keep = ~obs.ego_mask & (z < 45)
    keep[::2] = False; keep[:, ::2] = False
    P = cam.unproject(z)[keep]
    _, seg, t, lat, raw = rm.lookup(P, cam.pos, radius=50, max_abs=3.0)
    for s in np.unique(seg[seg >= 0]):
        m = seg == s
        samples[s].append(np.stack([t[m], lat[m], raw[m]], 1))
print("survey:", step, "steps,", used, "frames used")
resid, empty = [], []
for s, chunks in enumerate(samples):
    if not chunks or sum(len(c) for c in chunks) < 200:
        empty.append(s); continue
    X = np.concatenate(chunks)
    c = np.array([np.median(X[:, 2]), 0.0, 0.0])
    for tol in (0.6, 0.3, 0.2):                       # robust refit: keep inliers, refit plane
        r = X[:, 2] - (c[0] + c[1] * X[:, 0] + c[2] * X[:, 1])
        inl = np.abs(r) < tol
        if inl.sum() < 100: break
        A = np.stack([np.ones(inl.sum()), X[inl, 0], X[inl, 1]], 1)
        c = np.linalg.lstsq(A, X[inl, 2], rcond=None)[0]
    rm.plane[s] = c
    r = X[:, 2] - (c[0] + c[1] * X[:, 0] + c[2] * X[:, 1])
    resid.append([np.median(np.abs(r)), np.mean(np.abs(r) < 0.3), len(X)])
ok = [s for s in range(len(samples)) if s not in empty]
for s in empty:                                     # unobserved quads: copy the nearest surveyed one
    rm.plane[s] = rm.plane[min(ok, key=lambda o: min(abs(o - s), len(samples) - abs(o - s)))]
resid = np.array(resid)
print(f"quads {len(samples)}, unobserved {len(empty)}; offset c0 range {rm.plane[:,0].min():+.2f}..{rm.plane[:,0].max():+.2f}")
print(f"median |residual| {np.median(resid[:,0]):.3f} m; share of points within 0.3 m of fitted surface {np.average(resid[:,1], weights=resid[:,2]):.3f}")
rm.save(survey_path(track))
env.close()
