"""TensorRT inference plumbing: engine loading, buffers, and .onnx/.trt caching."""

import logging
import os
import time

import cv2 as cv
import numpy as np
import onnx
import tensorrt as trt
import torch
from onnxsim import simplify

logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# protobuf cap
SIMPLIFY_LIMIT_BYTES = 1_500_000_000

_TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


def build_engine(onnx_path, out_path, input_name, profile, fp16=False,
                 workspace_bytes=4 << 30):
    """Compile an ONNX model into a serialized TensorRT engine.

    Args:
        onnx_path: Path to the ONNX model.
        out_path: Where to write the .trt engine.
        input_name: Name of the input tensor ('images', 'pixel_values').
        profile: (min, opt, max), each a (batch, h, w) tuple.
        fp16: Allow FP16 kernels. I/O bindings stay FP32.
        workspace_bytes: Scratch memory TensorRT may use while picking kernels.
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(onnx_path):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    opt_profile = builder.create_optimization_profile()
    opt_profile.set_shape(input_name, *[(b, 3, h, w) for b, h, w in profile])
    config.add_optimization_profile(opt_profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build an engine from {onnx_path}")
    with open(out_path, "wb") as f:
        f.write(serialized)
    return out_path


def ensure_engine(name, directory, input_name, profile, export_onnx, fp16=False,
                  onnx_key=None):
    """Path to the cached engine for `name`, exporting and compiling if missing.

    Args:
        name: Model stem, used for both file names.
        directory: Where the .onnx and .trt live.
        input_name: Name of the ONNX input tensor.
        profile: (min, opt, max), each a (batch, h, w) tuple.
        export_onnx: Callable taking the output .onnx path.
        fp16: Allow FP16 kernels.
        onnx_key: ONNX cache key, when one export cannot serve every engine
            built from `name`. Defaults to `name`.
    """
    min_sz, opt_sz, max_sz = profile
    tag = f"{opt_sz[1]}x{opt_sz[2]}"
    if min_sz != max_sz:
        tag += f"_dyn{min_sz[1]}x{min_sz[2]}-{max_sz[1]}x{max_sz[2]}"
    if fp16:
        tag += "_fp16"
    trt_path = os.path.join(directory, f"{name}_{tag}.trt")
    if os.path.exists(trt_path):
        return trt_path

    onnx_dir = os.path.join(directory, f"{onnx_key or name}.onnx.d")
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    if not os.path.exists(onnx_path):
        os.makedirs(onnx_dir, exist_ok=True)
        logger.info(f"exporting ONNX: {name} -> {onnx_path}")
        export_onnx(onnx_path)
    logger.info(f"building TRT engine: {onnx_path} -> {trt_path}")
    return build_engine(onnx_path, trt_path, input_name, profile, fp16=fp16)


def export_onnx(module, dummy, output, input_name, output_names, dynamic_axes):
    """torch.onnx.export at opset 17, simplified with onnxsim when it is worth it."""
    torch.onnx.export(
        module,
        dummy,
        output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    # Measured off the module: a >2 GB export leaves the .onnx itself small.
    nbytes = 4 * sum(p.numel() for p in module.parameters())
    if nbytes > SIMPLIFY_LIMIT_BYTES:
        logger.info(f"{output} is {nbytes / 1e9:.1f} GB; keeping the raw export")
        return
    model, ok = simplify(onnx.load(output))
    if not ok:
        logger.warning(f"onnxsim failed for {output}; keeping the raw export")
        return
    onnx.save(model, output)


def letterbox(img_bgr, imgsz, stride=32, pad_value=(114, 114, 114)):
    """Aspect-preserving resize into an `imgsz` box, YOLO-style.

    Reproduces ultralytics' `LetterBox(imgsz, auto=True)`.
    `imgsz` bounds the output: an int, or an (h, w) tuple.

    Returns a (1, 3, h, w) float32 RGB array scaled to [0, 1].
    """
    ih, iw = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    h, w = img_bgr.shape[:2]
    scale = min(ih / h, iw / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    dw, dh = iw - nw, ih - nh
    if stride:
        dw, dh = dw % stride, dh % stride
    left, top = dw // 2, dh // 2
    resized = cv.resize(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB), (nw, nh))
    padded = cv.copyMakeBorder(
        resized, top, dh - top, left, dw - left, cv.BORDER_CONSTANT, value=pad_value
    )
    return np.transpose(np.array([padded], dtype=np.float32) / 255.0, (0, 3, 1, 2))


class TRTEngine:
    """A deserialized engine plus its I/O buffers.

    Buffers are allocated on the first inference and reused until the input shape
    changes, so a dynamic-profile engine costs a realloc per new size.

    Args:
        engine_path: Path to a serialized .trt engine.
        input_shape: Pre-allocate for this (b, 3, h, w); None defers to `_infer`.
        timing: Print a per-call stage breakdown.
    """

    def __init__(self, engine_path, input_shape=None, timing=False):
        self.timing = timing
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        names = [self.engine.get_tensor_name(i)
                 for i in range(self.engine.num_io_tensors)]
        self.input_name = names[0]
        self.all_output_names = [
            n for n in names
            if self.engine.get_tensor_mode(n) != trt.TensorIOMode.INPUT
        ]
        # Subclasses can filter this
        self.output_names = list(self.all_output_names)

        self._input_shape = None
        self.input_host = None
        self.input_dev = None
        self.output_tensors = {}
        if input_shape is not None:
            self._setup_buffers(tuple(input_shape))

    def _setup_buffers(self, input_shape):
        if input_shape == self._input_shape:
            return
        self._input_shape = input_shape
        self.context.set_input_shape(self.input_name, input_shape)

        dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(self.input_name)]
        self.input_host = torch.empty(input_shape, dtype=dtype).pin_memory()
        self.input_dev = torch.empty(input_shape, dtype=dtype, device="cuda")
        self.context.set_tensor_address(self.input_name, self.input_dev.data_ptr())

        # TRT needs an address for every output, even ones we never read.
        self.output_tensors = {}
        for name in self.all_output_names:
            tensor = torch.empty(
                tuple(self.context.get_tensor_shape(name)),
                dtype=_TRT_TO_TORCH[self.engine.get_tensor_dtype(name)],
                device="cuda",
            )
            self.output_tensors[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

    def _infer(self, inp):
        """Run the engine on a (b, 3, h, w) float32 array or CUDA tensor -> list of GPU tensors."""
        self._setup_buffers(tuple(inp.shape))
        if isinstance(inp, torch.Tensor):
            # inp may still be being written on the current stream
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self.input_dev.copy_(inp, non_blocking=True)
        else:
            self.input_host.copy_(torch.from_numpy(inp))
            with torch.cuda.stream(self.stream):
                self.input_dev.copy_(self.input_host, non_blocking=True)
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return [self.output_tensors[n] for n in self.output_names]

    def _report(self, label, stamps):
        """Print elapsed time between consecutive `stamps` for each stage."""
        if not self.timing:
            return
        stages = ("preprocess", "inference", "postprocess")
        parts = ", ".join(
            f"{s}: {b - a:.4f}s" for s, a, b in zip(stages, stamps, stamps[1:])
        )
        print(f"{f'[{label}]':<14} {parts}")

    @staticmethod
    def now():
        return time.time()

    def warmup(self, input_shape, iters=3):
        """Run `iters` dummy inferences so the first real call is not the slow one."""
        dummy = np.zeros(input_shape, dtype=np.float32)
        for _ in range(iters):
            self._infer(dummy)
