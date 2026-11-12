"""Thin wrapper around pystk2: one ego kart we control, built-in AI opponents."""
from dataclasses import dataclass, field
import numpy as np
import pystk2
from .camera import ChaseCamera

_initialized = False
SCALE = 2  # macOS retina: buffers come back at 2x the requested screen size


def init(width=320, height=240):
    global _initialized
    if not _initialized:
        g = pystk2.GraphicsConfig.hd()
        g.screen_width, g.screen_height = width, height
        pystk2.init(g)
        _initialized = True


@dataclass
class Obs:
    rgb: np.ndarray            # (2H, 2W, 3) uint8, what a camera would see
    speed: float               # forward speed, m/s (odometry)
    max_steer: float           # current max steering angle, rad (vehicle parameter)
    ego_mask: np.ndarray       # (H, W) bool: pixels covered by our own kart (calibration, like a hood mask)
    # ---- privileged simulator state: only the GT oracle and the metrics may read below ----
    depth_buf: np.ndarray = None   # (H, W) raw depth buffer
    cam: ChaseCamera = None
    kart: object = None
    others: list = field(default_factory=list)
    items: list = field(default_factory=list)
    time: float = 0.0
    racing: bool = False


class KartEnv:
    def __init__(self, track="lighthouse", n_karts=4, seed=0, laps=1, width=320, height=240,
                 ego_kart="tux", difficulty=2, ego_ai=False):
        init(width, height)
        self.W, self.H = width, height
        cfg = pystk2.RaceConfig(num_kart=n_karts, laps=laps, track=track, seed=seed, difficulty=difficulty)
        cfg.overlay = False
        # pystk2 attaches camera i to world kart i, but the player's kart is the *last* world kart.
        # Ask for one camera per kart and read the one that matches our kart id.
        cfg.num_cameras = n_karts
        p = cfg.players[0]
        p.kart = ego_kart
        p.camera_mode = pystk2.PlayerConfig.CameraMode.ON
        p.controller = (pystk2.PlayerConfig.Controller.AI_CONTROL if ego_ai
                        else pystk2.PlayerConfig.Controller.PLAYER_CONTROL)
        self.cfg, self.ego_ai = cfg, ego_ai
        self.race = pystk2.Race(cfg)
        self.state, self.track = pystk2.WorldState(), pystk2.Track()
        # engine max speed drives the camera distance: base 25 m/s x difficulty factor (medium-class kart)
        self.max_speed = 25.0 * {0: 0.6, 1: 0.8, 2: 0.92}.get(difficulty, 1.0)

    def reset(self):
        self.race.start()
        self.track.update()
        self.nodes = np.asarray(self.track.path_nodes).copy()
        self.widths = np.asarray(self.track.path_width)[:, 0].copy()
        self.length = float(self.track.length)
        self.cam = ChaseCamera(self.W, self.H, max_speed=self.max_speed, dt=self.cfg.step_size)
        return self.step(None)

    def step(self, action):
        if action is None or self.ego_ai:
            self.race.step()
        else:
            self.race.step(action)
        s = self.state
        s.update()
        me = s.players[0].kart
        self.cam.update(me.location, me.rotation, float(me.speed))
        rd = self.race.render_data[me.id]
        rgb = np.asarray(rd.image).copy()
        depth = np.asarray(rd.depth)[::SCALE, ::SCALE].copy()
        obs = Obs(rgb=rgb, speed=float(me.velocity_lc[2]), max_steer=float(me.max_steer_angle),
                  ego_mask=self._ego_mask(me), depth_buf=depth, cam=self.cam, kart=me,
                  others=[k for k in s.karts if k.id != me.id], items=list(s.items),
                  time=float(s.time), racing=int(s.phase.value) >= 7 if hasattr(s.phase, "value") else True)
        return obs

    def _ego_mask(self, me):
        """Screen-space box around our own kart, from the known camera and kart size."""
        from .camera import quat_to_mat
        R = quat_to_mat(np.asarray(me.rotation))
        w, h, l = np.asarray(me.size) * 0.5 + np.array([0.25, 0.0, 0.3])
        corners = np.array([[sx * w, sy, sz * l] for sx in (-1, 1) for sy in (-0.45, 0.75) for sz in (-1, 1)])
        u, v, z = self.cam.project(np.asarray(me.location) + corners @ R.T)
        m = np.zeros((self.H, self.W), bool)
        if (z > 0.2).all():
            u0, u1 = int(max(u.min(), 0)), int(min(u.max() + 1, self.W))
            v0, v1 = int(max(v.min(), 0)), int(min(v.max() + 1, self.H))
            m[v0:v1, u0:u1] = True
        return m

    def close(self):
        self.race.stop()
        del self.race
