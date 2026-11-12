"""Real-trained perception: off-the-shelf Cityscapes segmenter + monocular depth, classes remapped."""
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, AutoModelForDepthEstimation
from .scene import SceneMap, OTHER, ROAD, KART, OBSTACLE

# Cityscapes train ids: 0 road 1 sidewalk 2 building 3 wall 4 fence 5 pole 6 light 7 sign 8 vegetation
# 9 terrain 10 sky 11 person 12 rider 13 car 14 truck 15 bus 16 train 17 motorcycle 18 bicycle
REMAP = np.full(19, OTHER, np.uint8)
REMAP[0] = ROAD
REMAP[[13, 14, 15, 16, 17, 18, 12]] = KART
REMAP[[11, 5, 6, 7]] = OBSTACLE


class RealPerception:
    name = "real"

    def __init__(self, seg="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
                 depth="depth-anything/Depth-Anything-V2-Small-hf", out_hw=(240, 320), depth_scale=(1.0, 0.0)):
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.sp = AutoImageProcessor.from_pretrained(seg)
        self.seg = SegformerForSemanticSegmentation.from_pretrained(seg).to(self.dev).eval()
        self.dp = AutoImageProcessor.from_pretrained(depth)
        self.dm = AutoModelForDepthEstimation.from_pretrained(depth).to(self.dev).eval()
        self.out_hw, self.depth_scale = out_hw, depth_scale

    @torch.no_grad()
    def raw(self, rgb):
        """Returns (class map at out_hw, relative inverse depth at out_hw)."""
        im = Image.fromarray(rgb)
        x = self.sp(images=im, return_tensors="pt", size={"height": 512, "width": 672}).to(self.dev)
        logits = self.seg(**x).logits
        logits = torch.nn.functional.interpolate(logits, size=self.out_hw, mode="bilinear", align_corners=False)
        cls = REMAP[logits.argmax(1)[0].cpu().numpy()]
        y = self.dp(images=im, return_tensors="pt").to(self.dev)
        inv = self.dm(**y).predicted_depth[None]
        inv = torch.nn.functional.interpolate(inv, size=self.out_hw, mode="bilinear", align_corners=False)[0, 0]
        return cls, inv.cpu().numpy()

    def __call__(self, obs):
        cls, inv = self.raw(obs.rgb)
        a, b = self.depth_scale            # metric depth = 1 / (a * inv + b), calibrated on a held-out set
        return SceneMap(cls, (1.0 / np.maximum(a * inv + b, 1e-3)).astype(np.float32))
