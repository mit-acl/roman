"""TensorRT versions of the models ROMAN runs.

Each class takes the same weights the PyTorch path takes and caches its
exported .onnx and compiled .trt engine next to them, so nothing here needs a
separate download or conversion step.
"""

from roman.tensorrt.dinov2 import DINOv2TRT
from roman.tensorrt.engine import TRTEngine, build_engine, ensure_engine, letterbox
from roman.tensorrt.fastsam import FastSAMTRT
from roman.tensorrt.yolov8 import YOLOv8TRT

__all__ = [
    "DINOv2TRT",
    "FastSAMTRT",
    "TRTEngine",
    "YOLOv8TRT",
    "build_engine",
    "ensure_engine",
    "letterbox",
]
