"""Side-by-side debug video: camera frame | scene map | controller state."""
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from .scene import PALETTE, CLASS_NAMES

STATE_COLOR = dict(DRIVE=(60, 200, 90), STOP=(230, 50, 50), YIELD=(250, 190, 40), NO_ROAD=(200, 60, 220), RESCUE=(90, 160, 255))


def panel(obs, scene, state, ctrl):
    H, W = scene.classes.shape
    rgb = np.asarray(Image.fromarray(obs.rgb).resize((W, H), Image.BILINEAR))
    seg = (0.55 * PALETTE[scene.classes] + 0.45 * rgb).astype(np.uint8)
    d = np.clip(1.0 - np.log1p(scene.depth) / np.log1p(120.0), 0, 1)
    dep = np.stack([(255 * d).astype(np.uint8)] * 3, -1)
    im = Image.fromarray(np.concatenate([rgb, seg, dep], 1))
    dr = ImageDraw.Draw(im)
    c = STATE_COLOR.get(state, (255, 255, 255))
    dr.rectangle([W + 4, 4, W + 86, 22], fill=c)
    dr.text((W + 9, 8), state, fill=(0, 0, 0))
    dr.text((4, 4), f"{obs.speed * 3.6:4.0f} km/h  limit {ctrl.limit * 3.6:.0f}", fill=(255, 255, 255))
    dr.text((4, 16), f"t={obs.time:5.1f}s  next stop sign in {max(ctrl.next_stop - ctrl.odometer, 0):3.0f} m", fill=(255, 255, 255))
    for i, n in enumerate(CLASS_NAMES):
        dr.rectangle([W + 4 + 62 * i, H - 14, W + 12 + 62 * i, H - 6], fill=tuple(int(v) for v in PALETTE[i]))
        dr.text((W + 15 + 62 * i, H - 15), n, fill=(255, 255, 255))
    # aim marker
    u = W / 2 + np.tan(ctrl.aim) * ctrl.fx
    dr.line([(W + u, H * 0.42), (W + u, H * 0.52)], fill=(255, 255, 255), width=2)
    return np.asarray(im)


class Writer:
    def __init__(self, path, fps=20):
        self.w = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8)

    def add(self, frame):
        self.w.append_data(frame)

    def close(self):
        self.w.close()
