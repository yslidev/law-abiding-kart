"""Drive a lap with GT perception and save frames + GT labels: the held-out set for E2 and fine-tuning data."""
import sys, numpy as np
sys.path.insert(0, ".")
from lak.env import KartEnv
from lak.controller import LawAbidingController
from lak.perception_gt import GTPerception
track = sys.argv[1] if len(sys.argv) > 1 else "lighthouse"
every = int(sys.argv[2]) if len(sys.argv) > 2 else 15
env = KartEnv(track); obs = env.reset()
perc = GTPerception(env); ctrl = LawAbidingController(env.W, env.H)
rgb, cls, dep, ego = [], [], [], []
for step in range(3000):
    scene = perc(obs)
    a, st = ctrl.act(scene, obs, obs.racing)
    if step % every == 0 and obs.racing:
        rgb.append(obs.rgb); cls.append(scene.classes); dep.append(scene.depth.astype(np.float16)); ego.append(obs.ego_mask)
    if obs.kart.finished_laps >= 1: break
    obs = env.step(a)
np.savez_compressed(f"data/frames_{track}.npz", rgb=np.array(rgb), cls=np.array(cls), depth=np.array(dep), ego=np.array(ego))
print("saved", len(rgb), "frames", np.array(rgb).shape)
env.close()
