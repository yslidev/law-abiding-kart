"""Scripted speed profile to identify what the chase camera actually responds to."""
import sys, numpy as np, pystk2
sys.path.insert(0, ".")
from lak.env import KartEnv
from lak.camera import quat_to_mat
from scripts.measure_cam import measure
env = KartEnv("lighthouse", n_karts=1); obs = env.reset()
# (duration in steps, target speed m/s or 'rescue')
plan = [(60, 0), (60, 8), (60, 0), (60, 22), (60, 0), (40, 4), (5, "rescue"), (60, 0), (60, 15), (50, 0)]
sched = [p for n, p in plan for _ in range(n)]
rows = []
for step, tgt in enumerate(sched):
    me = obs.kart
    R = quat_to_mat(np.asarray(me.rotation))
    nxt = env.nodes[(me.node + 2) % len(env.nodes), 0]
    rel = (nxt - np.asarray(me.location)) @ R
    a = pystk2.Action(steer=float(np.clip(3 * np.arctan2(rel[0], rel[2]), -1, 1)))
    if tgt == "rescue": a.rescue = True
    else:
        a.acceleration = 1.0 if obs.speed < tgt - 0.3 else 0.0
        a.brake = bool(obs.speed > tgt + 1.0 and obs.speed > 0.3)
    h, p, r = measure(obs.cam.linear_depth(obs.depth_buf), obs.cam, env.H, env.W)
    rows.append((step, obs.time, obs.speed, h, p, r, obs.cam.offset[1], -obs.cam.offset[2]))
    obs = env.step(a)
rows = np.array(rows); np.save("out/cam_experiment.npy", rows)
print("step  time speed | height pitch rear | model y z")
for r in rows[::6]: print(f"{int(r[0]):4d} {r[1]:5.1f} {r[2]:5.1f} | {r[3]:.2f} {r[4]:5.1f} {r[5]:.2f} | {r[6]:.2f} {r[7]:.2f}")
env.close()
