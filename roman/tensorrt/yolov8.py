"""TensorRT YOLOv8 detection, used to mask out ignored/kept classes."""

import json
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

def _class_names(weights):
    """The checkpoint's class names, cached beside it so the .pt need not load."""
    cache = os.path.splitext(weights)[0] + "_names.json"
    if os.path.exists(cache):
        with open(cache) as f:
            return {int(i): n for i, n in json.load(f).items()}
    names = dict(YOLO(weights).names)
    with open(cache, "w") as f:
        json.dump(names, f)
    return names


def _export(weights):
    """Return an export_onnx callable for the YOLOv8 checkpoint at `weights`."""

    def run(onnx_path):
        model = YOLO(weights).model.eval().cuda()
        export_onnx(
            model,
            torch.zeros(1, 3, 640, 640, device="cuda"),
            onnx_path,
            input_name="images",
            output_names=["output0"],
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "output0": {0: "batch", 2: "anchors"},
            },
        )

    return run


class YOLOv8TRT(TRTEngine):
    """YOLOv8 detection on TensorRT, engine cached beside the .pt weights.

    Args:
        weights: Path to the YOLOv8 .pt checkpoint.
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

    def __init__(self, weights, directory=None, imgsz=640, conf=0.25, iou=0.7,
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

        self.names = _class_names(weights)
        self.conf = conf
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.output_names = [self.all_output_names[0]]

    def warmup(self, iters=3, input_shape=None):
        super().warmup(input_shape or (1, 3, *self.imgsz), iters)

    def detect(self, img_bgr):
        """Detect on a BGR image.

        Returns:
            An (N, 6) GPU tensor of [x1, y1, x2, y2, conf, class_id] in the
            original image's pixel coordinates.
        """
        t = [self.now()]
        inp = letterbox(img_bgr, self.imgsz)
        t.append(self.now())
        (pred,) = self._infer(inp)
        t.append(self.now())

        dets = ops.non_max_suppression(
            pred, self.conf, self.iou, agnostic=self.agnostic_nms,
            max_det=self.max_det,
        )[0]
        if len(dets):
            dets[:, :4] = ops.scale_boxes(inp.shape[2:], dets[:, :4], img_bgr.shape)
        t.append(self.now())
        self._report("YOLOv8_TRT", t)
        return dets
