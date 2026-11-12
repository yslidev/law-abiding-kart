"""Chase-camera model for SuperTuxKart.

pystk2 0.7.4 does not expose the camera matrices (the binding has that code
commented out), so we reproduce the engine's chase camera from the kart state.
The math below follows CameraNormal::moveCamera / positionCamera in the STK 1.5
source (src/graphics/camera/camera_normal.cpp), with the default user settings.

Conventions (Irrlicht, left-handed): camera x = right, y = up, z = forward.
Depth buffer is OpenGL depth in [0, 1] built from a D3D-style projection:
    d = 0.5 + 0.5 * f/(f-n) * (1 - n/z)
"""
import numpy as np

SMOOTH_POS = 0.2      # UserConfigParams::m_camera_forward_smooth_position
SMOOTH_ROT = 0.125    # UserConfigParams::m_camera_forward_smooth_rotation
CAM_DISTANCE = 1.0    # UserConfigParams::m_camera_distance
NEAR, FAR = 1.0, 1000.0
FOV_Y_DEG = 80.0      # UserConfigParams::m_camera_fov (vertical)


# ---------- quaternion helpers; q = [w, x, y, z] as pystk2 reports ----------
def quat_to_mat(q):
    """Rotation matrix whose columns are the kart's right/up/forward axes in world space.

    pystk2 reports the conjugate of the usual right-handed convention, so we
    conjugate first (checked against kart.front - kart.location).
    """
    w, x, y, z = q / np.linalg.norm(q)
    x, y, z = -x, -y, -z
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q1, q2, t):
    q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0:
        q2, dot = -q2, -dot
    if dot > 0.9995:
        q = q1 + t * (q2 - q1)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - t) * th) * q1 + np.sin(t * th) * q2) / np.sin(th)


class ChaseCamera:
    """Replays the engine's smoothed chase camera, one update per simulator step."""

    def __init__(self, width, height, max_speed=23.0, dt=0.1, fov_y_deg=FOV_Y_DEG,
                 near=NEAR, far=FAR):
        self.W, self.H = width, height
        self.max_speed, self.dt = max_speed, dt
        self.near, self.far = near, far
        self.h = 1.0 / np.tan(np.radians(fov_y_deg) / 2)   # focal, in NDC units (vertical)
        self.w = self.h / (width / height)                 # focal, in NDC units (horizontal)
        self.ready = False

    def reset(self, loc, rot):
        offset_z = -33.0 * np.sqrt(SMOOTH_POS)
        self.offset = np.array([0.0, 1.0, offset_z])
        self.q = np.asarray(rot, float).copy()
        self.up = quat_to_mat(self.q)[:, 1]
        self.ready = True

    def update(self, loc, rot, speed):
        """Advance one step. Returns (position, R) with R rows = camera right, up, forward."""
        loc, rot = np.asarray(loc, float), np.asarray(rot, float)
        if not self.ready:
            self.reset(loc, rot)
        ratio = max(speed / self.max_speed, -0.12)
        cam_dist = (-2.8 - 5.6 * ratio) * np.sqrt(SMOOTH_POS)
        cam_dist = min(cam_dist, -2.0 * CAM_DISTANCE)
        wanted = np.array([0.0, 0.85 + ratio / 2.5, cam_dist])
        delta = np.clip(self.dt / max(self.dt, SMOOTH_POS), 0, 1)
        delta2 = np.clip(self.dt / max(self.dt, SMOOTH_ROT), 0, 1)

        q_prev = self.q
        self.q = slerp(q_prev, rot, delta2)
        Rk = quat_to_mat(q_prev)                       # engine uses the *previous* smoothed rotation
        pos = loc + Rk @ self.offset
        self.offset = self.offset + (wanted - self.offset) * delta
        target = loc + Rk @ np.array([0.0, 0.5, 0.0])

        kart_up = quat_to_mat(rot)[:, 1]
        self.up = 0.04 * kart_up + 0.96 * self.up

        z = target - pos
        z /= np.linalg.norm(z)
        x = np.cross(self.up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        self.pos, self.R = pos, np.stack([x, y, z])
        return self.pos, self.R

    # ---------- geometry ----------
    def linear_depth(self, d):
        """Depth-buffer value -> metric distance along the optical axis."""
        n, f = self.near, self.far
        return n / np.maximum(1.0 - (2.0 * d - 1.0) * (f - n) / f, 1e-6)

    def project(self, pts):
        """World points (N,3) -> pixel (u, v) and camera-space depth z."""
        c = (np.asarray(pts, float) - self.pos) @ self.R.T
        z = c[:, 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = (c[:, 0] * self.w / zs + 1) * 0.5 * self.W
        v = (1 - c[:, 1] * self.h / zs) * 0.5 * self.H
        return u, v, z

    def unproject(self, z):
        """Metric depth map (H,W) -> world points (H,W,3)."""
        H, W = z.shape
        u = (np.arange(W) + 0.5) / W * 2 - 1
        v = 1 - (np.arange(H) + 0.5) / H * 2
        xc = u[None, :] / self.w * z
        yc = v[:, None] / self.h * z
        cam = np.stack([xc, yc, z], -1)
        return cam @ self.R + self.pos
