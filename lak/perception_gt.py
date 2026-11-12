"""GT (oracle) perception.

pystk2's instance buffer is not usable (it returns an unrelated render target), so
the oracle labels pixels geometrically: unproject the engine depth buffer to world
points with the reconstructed camera, then test each point against a surveyed map
of the road surface (ROAD), other karts' boxes (KART) and item positions (OBSTACLE).

The road map comes from the engine's driveline quads. Those quads float above the
visible surface by a varying amount, so `scripts/survey_track.py` drives one lap
and fits, per quad, a plane   height_above_driveline = c0 + c1*t + c2*lat
(t = position along the quad, lat = signed lateral offset in metres).
"""
import os
import numpy as np
from .scene import SceneMap, OTHER, ROAD, KART, OBSTACLE
from .camera import quat_to_mat


class RoadMap:
    def __init__(self, nodes, widths):
        a, b = nodes[:, 0], nodes[:, 1]
        ab = b - a
        L = np.sqrt(ab[:, 0] ** 2 + ab[:, 2] ** 2) + 1e-9
        self.a, self.ab, self.L = a, ab, L
        self.side = np.stack([ab[:, 2], -ab[:, 0]], 1) / L[:, None]
        self.half_w = widths / 2
        self.mid = (a + b) / 2
        self.seg_r = np.linalg.norm(ab, axis=1) / 2 + self.half_w
        self.plane = np.zeros((len(a), 3))      # c0, c1, c2 per quad
        self.plane[:, 0] = -0.9

    def lookup(self, P, cam_pos, radius=150.0, max_abs=3.0):
        """For world points P (N,3): residual height above the surveyed road surface (inf if not over a
        quad), plus the quad index, t and lat of the best match."""
        res = np.full(len(P), np.inf)
        seg = np.full(len(P), -1)
        tt = np.zeros(len(P)); ll = np.zeros(len(P)); raw = np.zeros(len(P))
        near = np.where(np.linalg.norm(self.mid - cam_pos, axis=1) < radius + self.seg_r)[0]
        for s in near:
            rx, rz = P[:, 0] - self.a[s, 0], P[:, 2] - self.a[s, 2]
            t = (rx * self.ab[s, 0] + rz * self.ab[s, 2]) / self.L[s] ** 2
            lat = rx * self.side[s, 0] + rz * self.side[s, 1]
            idx = np.where((t >= -0.02) & (t <= 1.02) & (np.abs(lat) <= self.half_w[s]))[0]
            if not len(idx):
                continue
            dy = P[idx, 1] - (self.a[s, 1] + t[idx] * self.ab[s, 1])
            r = dy - (self.plane[s, 0] + self.plane[s, 1] * t[idx] + self.plane[s, 2] * lat[idx])
            better = (np.abs(r) < np.abs(res[idx])) & (np.abs(dy) < max_abs)
            j = idx[better]
            res[j], seg[j], tt[j], ll[j], raw[j] = r[better], s, t[idx][better], lat[idx][better], dy[better]
        return res, seg, tt, ll, raw

    def save(self, path):
        np.save(path, self.plane)

    def load(self, path):
        self.plane = np.load(path)


def survey_path(track):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"survey_{track}.npy")


class GTPerception:
    name = "gt"

    def __init__(self, env, road_tol=0.3):
        self.map = RoadMap(env.nodes, env.widths)
        path = survey_path(env.cfg.track)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no road survey for {env.cfg.track}: run scripts/survey_track.py {env.cfg.track}")
        self.map.load(path)
        self.road_tol = road_tol

    def __call__(self, obs):
        cam = obs.cam
        z = cam.linear_depth(obs.depth_buf).astype(np.float32)
        H, W = z.shape
        P = cam.unproject(z).reshape(-1, 3)
        cls = np.full(H * W, OTHER, np.uint8)
        res, *_ = self.map.lookup(P, cam.pos)
        dist = z.reshape(-1)
        # The reconstructed camera can be off by a fraction of a degree in pitch. That shows up as a
        # residual that grows linearly with distance, so fit and remove  res ~ a + b*dist  each frame.
        ok = np.isfinite(res) & (np.abs(res) < 0.25 + 0.02 * dist) & (dist < 80)
        if ok.sum() > 800:
            a, b = 0.0, 0.0
            for _ in range(2):
                inl = ok & (np.abs(res - (a + b * dist)) < 0.25)
                if inl.sum() < 400:
                    break
                b, a = np.polyfit(dist[inl], res[inl], 1)
            if abs(a) < 0.4 and abs(b) < 0.03:
                res = res - (a + b * dist)
        cls[(np.abs(res) < self.road_tol + 0.003 * dist) & (dist < 400)] = ROAD
        for it in obs.items:
            m = (np.linalg.norm(P - np.asarray(it.location), axis=1) < 0.95) & (cls != ROAD)
            cls[m] = OBSTACLE
        for k in obs.others:
            R = quat_to_mat(np.asarray(k.rotation))
            loc = (P - np.asarray(k.location)) @ R
            w, h, l = np.asarray(k.size)
            m = (np.abs(loc[:, 0]) < w / 2 + 0.15) & (np.abs(loc[:, 2]) < l / 2 + 0.15) & \
                (loc[:, 1] > -0.6) & (loc[:, 1] < h + 0.5) & (cls != ROAD)
            cls[m] = KART
        return SceneMap(cls.reshape(H, W), z)
