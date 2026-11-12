# Law-Abiding Kart

CS 180 final project. Spec: https://claude.ai/artifact/TM4Moo94U9C22D3sQHRtDb

A fixed, rule-following controller drives a SuperTuxKart race; only the perception
variant changes. Status as of 2026-11-10: the simulator, GT (oracle) perception,
controller, metrics and video work end to end on one track. The real-trained
variant is scored offline (E2) but not yet driven in closed loop.

## Run

```bash
./run.sh scripts/survey_track.py lighthouse                     # once per track: road-surface map for GT
./run.sh scripts/run_episode.py --video out/lap.mp4             # law-abiding lap, GT perception
./run.sh scripts/run_episode.py --lawless --tag _lawless        # E3: rules off
./run.sh scripts/run_episode.py --ego-ai --tag _ai              # E3: built-in AI drives
./run.sh scripts/dump_frames.py lighthouse 15                   # save frames + GT labels
.venv/bin/python scripts/eval_perception.py lighthouse          # E2: real-trained models vs GT
```

Use `./run.sh` for anything that starts the simulator: SuperTuxKart hangs at init
when the Mac's display is asleep, so the wrapper wakes it and keeps it on.

## Layout

- `lak/env.py` — pystk2 wrapper; one controlled kart, built-in AI opponents
- `lak/camera.py` — chase-camera reconstruction (pystk2 does not expose it)
- `lak/perception_gt.py` — geometric oracle: depth buffer + camera + surveyed road map
- `lak/perception_real.py` — SegFormer (Cityscapes) + Depth Anything V2, remapped to 4 classes
- `lak/controller.py` — the law-abiding rules; reads only the scene map and odometry
- `lak/scene.py` — the scene-map contract (OTHER, ROAD, KART, OBSTACLE + metric depth)
- `scripts/measure_cam.py`, `scripts/cam_experiment.py` — checks of the camera model

## Things that differ from the spec

- **pystk2 0.7.4 on Python 3.12**, not `pystk` on 3.9/3.10. The old package cannot
  install any more: its asset server is gone.
- **No instance buffer.** pystk2's `instance` returns an unrelated render target, so GT
  labels are computed geometrically. ROAD is the engine's drivable area, which is wider
  than the painted path. OBSTACLE is items only (boxes, bananas, nitro, gum).
- **No camera matrices.** The camera is re-derived from kart state using the engine's
  own formula (`camera_normal.cpp`); it matches depth-based measurements of camera
  height to a few centimetres. A per-frame linear correction removes the remaining
  sub-degree pitch error.
- **Camera bug in pystk2:** camera *i* follows world kart *i*, but the player's kart is
  the last one. The wrapper requests one camera per kart and reads ours.
- Kart name tags ("Tux") are drawn into both RGB and depth and cannot be turned off.

## First results (lighthouse, seed 0, 3 opponents — one run each)

| Driver | Lap time (s) | Place of 4 | Stops | Lost-road events |
| --- | --- | --- | --- | --- |
| Law-abiding, GT perception | 111.9 | 4 | 5 | 0 |
| Same steering, rules off | 41.8 | 4 | 0 | 0 |
| Built-in AI | 35.8 | 2 | 0 | 0 |

Real-trained models vs GT on 73 frames: IoU ROAD 0.755, OTHER 0.789, KART 0.002,
OBSTACLE 0.011 (mIoU 0.389); depth AbsRel 0.45 with one affine calibration;
232 ms per frame on the M4's GPU.

## Not done yet

Real-trained closed loop, kart-tuned fine-tuning, stylized variant, other tracks,
multi-seed sweeps (E1), ablation (E4), charts, highlight reel. The missed/phantom
yield counters are implemented but not validated.
