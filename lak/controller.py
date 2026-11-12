"""The law-abiding controller. Held fixed across all perception variants.

It reads only the scene map (class map + depth), the ego mask, and odometry
(speed, max steering angle). It never sees simulator state.
"""
from dataclasses import dataclass
import numpy as np
import pystk2
from .scene import ROAD, KART, OBSTACLE

DRIVE, STOP, YIELD, NO_ROAD, RESCUE = "DRIVE", "STOP", "YIELD", "NO_ROAD", "RESCUE"


@dataclass
class Params:
    speed_limit: float = 10.0        # m/s, far below the kart's ~23 m/s top speed
    curve_speed: float = 6.0         # m/s when the road ahead bends
    curve_angle: float = 0.22        # rad of aim angle that counts as a curve
    stop_every: float = 150.0        # metres between imaginary stop signs
    stop_hold: float = 1.5           # seconds of full stop
    follow_time: float = 1.5         # seconds of headway to keep
    follow_min: float = 6.0          # metres, minimum gap
    corridor: float = 1.6            # metres, half-width of our lane ahead
    look_near: float = 7.0           # metres from camera: look-ahead band for lane keeping
    look_far: float = 22.0
    steer_gain: float = 2.2
    min_road_frac: float = 0.03      # fraction of look-ahead band pixels that must be ROAD
    lost_patience: float = 4.0       # seconds stopped with no road before calling for a tow
    stop_signs: bool = True
    follow: bool = True
    limit: bool = True


class LawAbidingController:
    def __init__(self, width, height, fov_y_deg=80.0, dt=0.1, params=None):
        self.p = params or Params()
        self.W, self.H, self.dt = width, height, dt
        self.fy = 0.5 * height / np.tan(np.radians(fov_y_deg) / 2)   # focal length in pixels
        self.fx = self.fy
        self.xs = (np.arange(width) + 0.5 - width / 2) / self.fx      # tan of horizontal angle per column
        self.odometer = 0.0
        self.next_stop = self.p.stop_every
        self.stop_timer = 0.0
        self.lost_timer = 0.0
        self.rescue_timer = 0.0
        self.steer = 0.0
        self.log = dict(stops=0, lost_events=0, rescues=0)
        self._was_lost = False

    def act(self, scene, obs, racing=True):
        p, cls, z = self.p, scene.classes, scene.depth
        speed = obs.speed
        self.odometer += max(speed, 0) * self.dt
        a = pystk2.Action()            # drift / nitro / fire stay False forever
        x = self.xs[None, :] * z       # lateral position of every pixel, metres (camera frame)

        # ---- lane keeping: aim at the centroid of ROAD in the look-ahead band ----
        band = (z > p.look_near) & (z < p.look_far)
        road = band & (cls == ROAD)
        n_band, n_road = int(band.sum()), int(road.sum())
        lost = n_road < max(p.min_road_frac * max(n_band, 1), 30)
        aim = 0.0
        if not lost:
            wgt = z[road] ** 2                      # undo perspective: equal weight per square metre of road
            cx, cz = np.average(x[road], weights=wgt), np.average(z[road], weights=wgt)
            aim = float(np.arctan2(cx, cz))
            target = np.clip(p.steer_gain * aim / max(obs.max_steer, 0.1), -1, 1)
            self.steer += 0.6 * (target - self.steer)
        a.steer = float(self.steer)

        # ---- following distance: any KART in our corridor closer than the headway gap ----
        gap = max(p.follow_min, p.follow_time * max(speed, 0)) + 3.0   # +3 m: camera sits behind our kart
        ahead = (cls == KART) & ~obs.ego_mask & (np.abs(x) < p.corridor) & (z < gap) & (z > 1.0)
        kart_close = p.follow and int(ahead.sum()) >= 12
        self.gap_seen = float(z[ahead].min()) if kart_close else np.inf

        # ---- speed limit ----
        limit = p.speed_limit if p.limit else 99.0
        if p.limit and abs(aim) > p.curve_angle:
            limit = p.curve_speed

        # ---- state machine ----
        state = DRIVE
        if self.rescue_timer > 0:
            self.rescue_timer -= self.dt
            state = RESCUE
        elif p.stop_signs and racing and (self.stop_timer > 0 or self.odometer >= self.next_stop):
            state = STOP
            if self.stop_timer <= 0 and self.odometer >= self.next_stop:
                self.stop_timer = -1.0                     # braking phase
            if abs(speed) < 0.3 and self.stop_timer < 0:
                self.stop_timer = p.stop_hold             # full stop reached: start the hold
                self.log["stops"] += 1
            elif self.stop_timer > 0:
                self.stop_timer -= self.dt
                if self.stop_timer <= 0:
                    self.stop_timer = 0.0
                    self.next_stop = self.odometer + p.stop_every
                    state = DRIVE
        elif lost and racing:
            state = NO_ROAD
        elif kart_close:
            state = YIELD

        if state == NO_ROAD:
            if not self._was_lost:
                self.log["lost_events"] += 1
            self.lost_timer += self.dt
            if self.lost_timer > p.lost_patience and abs(speed) < 0.5:
                a.rescue = True                            # the law-abiding move: call a tow truck
                self.log["rescues"] += 1
                self.lost_timer, self.rescue_timer = 0.0, 3.0
                state = RESCUE
        else:
            self.lost_timer = max(0.0, self.lost_timer - self.dt)
        self._was_lost = state == NO_ROAD

        if state == DRIVE:
            a.acceleration = 1.0 if speed < limit - 0.3 else 0.0
            a.brake = speed > limit + 1.5
        elif state in (STOP, YIELD, NO_ROAD):
            a.acceleration = 0.15 if speed < -0.2 else 0.0   # don't roll backwards
            a.brake = speed > 0.3
        self.state, self.aim, self.limit = state, aim, limit
        return a, state
