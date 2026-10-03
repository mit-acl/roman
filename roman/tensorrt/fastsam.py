"""TensorRT FastSAM: BGR image in, instance masks out."""

import os

import torch
from ultralytics import YOLO
from ultralytics.yolo.utils import ops

from roman.tensorrt.engine import (
    TRTEngine,
    ensure_engine,
    export_onnx,
    letterbox,
)


def _export(weights):
    """Return an export_onnx callable for the FastSAM checkpoint at `weights`."""

    def run(onnx_path):
        model = YOLO(weights).model.eval().cuda()
        export_onnx(
            model,
            torch.zeros(1, 3, 256, 256, device="cuda"),
            onnx_path,
            input_name="images",
            output_names=["output0", "output1"],
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "output0": {0: "batch", 1: "anchors"},
                "output1": {0: "batch", 2: "mask_height", 3: "mask_width"},
            },
        )

    return run


class FastSAMTRT(TRTEngine):
    """FastSAM on TensorRT, with the engine cached beside the .pt weights.

    Args:
        weights: Path to the FastSAM .pt checkpoint.
        directory: Where the cached .onnx and .trt live. Defaults to the
            directory holding `weights`.
        imgsz: Box the letterboxed input is bounded by, as an int or (h, w).
        conf: Confidence threshold for NMS.
        iou: IoU threshold for NMS.
        agnostic_nms: Class-agnostic NMS.
        max_det: Cap on detections per image.
        fp16: Allow FP16 kernels.
        timing: Print a per-call stage breakdown.
    """

    def __init__(self, weights, directory=None, imgsz=256, conf=0.25, iou=0.7,
                 agnostic_nms=False, max_det=100, fp16=False, timing=False):
        name = os.path.splitext(os.path.basename(weights))[0]
        directory = directory or os.path.dirname(os.path.abspath(weights))
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
        engine_path = ensure_engine(
            name, directory, "images",
            ((1, 64, 64), (1, *self.imgsz), (1, *self.imgsz)),
            _export(weights), fp16=fp16,
        )
        super().__init__(engine_path, timing=timing)

        self.conf = conf
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det

        # Pick prototypes out by shape
        self._setup_buffers((1, 3, *self.imgsz))
        proto = max(
            (n for n, t in self.output_tensors.items() if t.ndim == 4),
            key=lambda n: self.output_tensors[n].shape[-2:].numel(),
        )
        self.output_names = [self.all_output_names[0], proto]

    def warmup(self, iters=3, input_shape=None):
        super().warmup(input_shape or (1, 3, *self.imgsz), iters)

    def segment(self, img):
        """Segment an image, as a drop-in for `FastSAM(img)`.

        Takes the same array the ultralytics predictor takes and applies the same
        channel flip; note that it reads a numpy input as BGR whatever it holds.

        Returns:
            A (N, H, W) GPU mask tensor at the original image resolution, or
            None when nothing was detected.
        """
        t = [self.now()]
        inp = letterbox(img, self.imgsz)
        t.append(self.now())
        det, proto = self._infer(inp)
        t.append(self.now())

        preds = ops.non_max_suppression(
            det, self.conf, self.iou, agnostic=self.agnostic_nms,
            max_det=self.max_det, nc=1,
        )[0]
        masks = None
        if len(preds):
            # NMS boxes are letterboxed; masks are cut at the original resolution.
            preds[:, :4] = ops.scale_boxes(
                inp.shape[2:], preds[:, :4], img.shape
            )
            masks = ops.process_mask_native(
                proto[0], preds[:, 6:], preds[:, :4], img.shape[:2]
            )
        t.append(self.now())
        self._report("FastSAM_TRT", t)
        return masks
