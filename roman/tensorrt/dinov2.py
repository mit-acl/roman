"""TensorRT DINOv2 patch features, for the semantics and frame-descriptor paths."""

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from roman.tensorrt.engine import TRTEngine, ensure_engine, export_onnx

PATCH_SIZE = 14  # every DINOv2 variant
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr, imgsz):
    """Resize and ImageNet-normalize a BGR image into a (1, 3, H, W) float32 CUDA tensor.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        imgsz: int to scale the short side to, or an (h, w) tuple for an exact
            resize.
    """
    if isinstance(imgsz, int):
        h, w = img_bgr.shape[:2]
        scale = imgsz / min(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
    else:
        new_h, new_w = imgsz
    img = torch.from_numpy(np.ascontiguousarray(img_bgr)).cuda()
    resized = img.flip(-1).permute(2, 0, 1)[None].float()
    # PIL resizes horizontally then vertically, rounding to uint8 after each pass
    for size in [(resized.shape[2], new_w), (new_h, new_w)]:
        resized = torch.nn.functional.interpolate(
            resized, size=size, mode="bicubic", align_corners=False, antialias=True
        ).round_().clamp_(0, 255)
    mean = torch.as_tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
    return (resized / 255.0 - mean) / std


def reshape_patches(last_hidden_state, input_hw):
    """Drop the CLS token and fold the patch tokens into a (1, h, w, D) grid."""
    h = input_hw[0] // PATCH_SIZE
    w = input_hw[1] // PATCH_SIZE
    return last_hidden_state[:, 1:, :].reshape(1, h, w, -1)


def default_imgsz(model_name):
    """The short side `model_name`'s own image processor resizes to."""
    size = AutoImageProcessor.from_pretrained(model_name).size
    if "shortest_edge" in size:
        return size["shortest_edge"]
    return min(size["height"], size["width"])


def _export(model_name):
    """Return an export_onnx callable for the HuggingFace model `model_name`."""

    def run(onnx_path):
        class Wrapper(torch.nn.Module):
            """A single positional tensor in, last_hidden_state out."""

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values):
                return self.model(pixel_values=pixel_values).last_hidden_state

        # SDPA has no opset-17 lowering, so export from the eager attention.
        model = AutoModel.from_pretrained(
            model_name, attn_implementation="eager"
        ).eval().cuda()
        export_onnx(
            Wrapper(model),
            torch.zeros(1, 3, 256, 256, device="cuda"),
            onnx_path,
            input_name="pixel_values",
            output_names=["last_hidden_state"],
            dynamic_axes={
                "pixel_values": {0: "batch", 2: "height", 3: "width"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
        )

    return run


class DINOv2TRT(TRTEngine):
    """DINOv2 on TensorRT, with a dynamic-resolution engine.

    Args:
        model_name: HuggingFace model id, e.g. 'facebook/dinov2-base'.
        directory: Where the cached .onnx and .trt live.
        imgsz: Short side the input is scaled to. Defaults to the one the
            model's own image processor uses, which is what the PyTorch path
            feeds it.
        max_scale: Largest supported side, as a multiple of `imgsz`.
        fp16: Allow FP16 kernels.
        timing: Print a per-call stage breakdown.
    """

    def __init__(self, model_name, directory, imgsz=None, max_scale=3,
                 fp16=False, timing=False):
        name = model_name.split("/")[-1]
        imgsz = default_imgsz(model_name) if imgsz is None else imgsz
        engine_path = ensure_engine(
            name, directory, "pixel_values",
            (
                (1, imgsz, imgsz),
                (1, imgsz, 2 * imgsz),
                (1, max_scale * imgsz, max_scale * imgsz),
            ),
            _export(model_name), fp16=fp16,
        )
        super().__init__(engine_path, timing=timing)
        self.imgsz = imgsz

    def warmup(self, iters=3, input_shape=None):
        super().warmup(input_shape or (1, 3, self.imgsz, 2 * self.imgsz), iters)

    def embed(self, img_bgr, reshape=False):
        """Patch features for a BGR image.

        Returns:
            (1, num_patches + 1, D) with the CLS token, or (1, h, w, D) with it
            dropped when `reshape` is set.
        """
        t = [self.now()]
        inp = preprocess(img_bgr, self.imgsz)
        t.append(self.now())
        (out,) = self._infer(inp)
        t.append(self.now())
        if reshape:
            out = reshape_patches(out, inp.shape[2:])
        t.append(self.now())
        self._report("DINOv2_TRT", t)
        return out
