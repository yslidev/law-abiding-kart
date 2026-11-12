"""Run one race with a given perception variant and the fixed law-abiding controller."""
import argparse, json, sys, time
import numpy as np
sys.path.insert(0, ".")
from lak.env import KartEnv
from lak.controller import LawAbidingController, Params, DRIVE, STOP, YIELD, NO_ROAD
from lak.perception_gt import GTPerception
from lak.camera import quat_to_mat
from lak import viz

ap = argparse.ArgumentParser()
ap.add_argument("--track", default="lighthouse")
ap.add_argument("--perception", default="gt")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--karts", type=int, default=4)
ap.add_argument("--max-steps", type=int, default=4000)
ap.add_argument("--video", default="")
ap.add_argument("--lawless", action="store_true", help="turn the law-abiding rules off (E3 baseline)")
ap.add_argument("--ego-ai", action="store_true", help="let the built-in AI drive our kart (E3 baseline)")
ap.add_argument("--tag", default="")
args = ap.parse_args()

env = KartEnv(args.track, n_karts=args.karts, seed=args.seed, ego_ai=args.ego_ai)
obs = env.reset()
perc = {"gt": GTPerception}[args.perception](env)
params = Params(stop_signs=False, follow=False, limit=False) if args.lawless else Params()
ctrl = LawAbidingController(env.W, env.H, dt=env.cfg.step_size, params=params)
writer = viz.Writer(args.video) if args.video else None

rows, phantom, missed, prev_ph, prev_mi = [], 0, 0, False, False
t0, perc_ms = time.time(), []
finished = False
for step in range(args.max_steps):
    t = time.time(); scene = perc(obs); perc_ms.append((time.time() - t) * 1e3)
    action, state = ctrl.act(scene, obs, racing=obs.racing)
    # ---- ground truth for the yield metrics: is a kart really inside our headway gap? ----
    me = obs.kart
    R = quat_to_mat(np.asarray(me.rotation))
    gap = max(ctrl.p.follow_min, ctrl.p.follow_time * max(obs.speed, 0))
    rel = [(np.asarray(k.location) - np.asarray(me.location)) @ R for k in obs.others]
    fwd = [r[2] - 1.5 for r in rel if abs(r[0]) < ctrl.p.corridor and r[2] > 0]     # bumper-to-bumper-ish
    truly_close = any(f < 0.7 * gap for f in fwd)
    nobody_near = not any(f < 1.5 * gap for f in fwd)
    ph, mi = state == YIELD and nobody_near, truly_close and state == DRIVE
    phantom += ph and not prev_ph; missed += mi and not prev_mi
    prev_ph, prev_mi = ph, mi
    rows.append(dict(t=obs.time, dist=float(me.distance_down_track), speed=obs.speed, state=state,
                     steer=float(action.steer), on_road=bool(me.is_on_road)))
    if writer:
        writer.add(viz.panel(obs, scene, state, ctrl))
    if me.finished_laps >= 1 or me.has_finished_race:
        finished = True
        break
    obs = env.step(action)

me = obs.kart
states = [r["state"] for r in rows]
res = dict(track=args.track, perception=args.perception, seed=args.seed, lawless=args.lawless, ego_ai=args.ego_ai,
           finished=finished, steps=len(rows), sim_time=round(rows[-1]["t"], 2),
           lap_time=round(float(me.finish_time), 2) if finished and me.finish_time > 0 else None,
           place=int(me.position), progress=round(max(r["dist"] for r in rows) / env.length, 3),
           mean_speed=round(float(np.mean([r["speed"] for r in rows if r["t"] > 0])), 2),
           stops=ctrl.log["stops"], lost_events=ctrl.log["lost_events"], rescues=ctrl.log["rescues"],
           phantom_yields=int(phantom), missed_yields=int(missed),
           frac={s: round(states.count(s) / len(states), 3) for s in (DRIVE, STOP, YIELD, NO_ROAD)},
           off_road_frac=round(1 - float(np.mean([r["on_road"] for r in rows])), 3),
           perception_ms=round(float(np.mean(perc_ms)), 1), wall_s=round(time.time() - t0, 1))
print(json.dumps(res))
if writer:
    writer.close()
name = f"out/run_{args.track}_{args.perception}_{args.seed}{args.tag}"
json.dump(dict(result=res, rows=rows), open(name + ".json", "w"))
env.close()
