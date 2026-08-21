#!/usr/bin/env python3
"""GPU backend abstraction for face detection.

Provides a uniform ``DetectorBackend`` protocol with one implementation per
GPU vendor. Each implementation exposes the same ``detect(frame_bgr)`` method
and returns boxes in the same ``(x, y, w, h, conf)`` tuple layout used by
``tools/annotate_all_faces.py`` — this guarantees that swapping CPU ↔ GPU
does not change downstream consumers.

Why this layer exists
---------------------
We want one ``detect_gpu.py`` script that runs on a developer Mac today
(CoreML / OpenCL over Metal) and on a Linux box with an Ascend NPU tomorrow,
without rewriting the calling code. Adding a new vendor is one new class —
it never touches the script.

| Backend class        | Vendor           | Hook                                            |
|----------------------|------------------|-------------------------------------------------|
| ``CPUBackend``       | any host         | cv2.dnn Res10-SSD on CPU (ground-truth baseline)|
| ``OpenCLBackend``    | Intel/AMD/NVIDIA | cv2.dnn with DNN_TARGET_OPENCL                  |
| ``VulkanBackend``    | AMD/NVIDIA       | cv2.dnn with DNN_TARGET_VULKAN                  |
| ``CoreMLBackend``    | Apple ANE/GPU    | onnxruntime CoreML EP (Apple Silicon)           |
| ``CUDABackend``      | NVIDIA           | onnxruntime CUDA EP (cuDNN)                     |
| ``ROCmBackend``      | AMD (Linux)      | onnxruntime ROCm EP                             |
| ``DirectMLBackend``  | AMD/NVIDIA (Win) | onnxruntime DirectML EP                         |
| ``ACLBackend``       | Huawei Ascend    | onnxruntime ACL EP (CANN)                       |
| ``MLUBackend``       | Cambricon        | onnxruntime Cambricon EP                        |
| ``AutoBackend``      | dispatcher       | picks the first available GPU backend            |

All non-cv2 backends load the SAME ``res10_ssd.onnx`` model (produced by
``tools/convert_res10_to_onnx.py``) so boxes match the CPU baseline by
construction. cv2-based backends load the same Caffe weights via
``cv2.dnn.readNetFromCaffe``.

Bit-identical results
---------------------
Because the model is identical and the input blob is identical
(``cv2.dnn.blobFromImage(mean=(104,177,123), scale=1.0, size=300x300)``),
the same boxes are returned within float32 precision. The
``tools/compare_cpu_gpu.py`` helper verifies IoU ≥ 0.95 across videos.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Optional

import cv2
import numpy as np

# Default Res10 SSD model paths. Override via env if you have a custom
# copy elsewhere; the Caffe proto/model pair is the original open-source
# release by davisking (https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector).
CAFFE_PROTO_DEFAULT = Path("/tmp/deploy.prototxt.txt")
CAFFE_MODEL_DEFAULT = Path("/tmp/res10_300x300_ssd_iter_140000.caffemodel")
ONNX_MODEL_DEFAULT = Path("/tmp/res10_ssd.onnx")

DNN_INPUT_SIZE = 300
DNN_MEAN = (104.0, 177.0, 123.0)
DNN_SCALE = 1.0

Box = tuple[int, int, int, int, float]  # x, y, w, h, confidence


@dataclass(frozen=True)
class BackendInfo:
    name: str
    vendor: str
    device: str
    detail: str


class DetectorBackend(Protocol):
    """Common interface for any face-detection backend (CPU or GPU)."""

    name: str
    vendor: str

    def info(self) -> BackendInfo: ...
    def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.5) -> list[Box]: ...


# ---------- shared preprocessing ----------

def blob_from_bgr(frame_bgr: np.ndarray,
                  size: int = DNN_INPUT_SIZE) -> np.ndarray:
    """Mimics ``cv2.dnn.blobFromImage(..., scale=1.0, size=(s,s), mean=...)``
    so that all backends receive an identical tensor."""
    resized = cv2.resize(frame_bgr, (size, size))
    arr = resized.astype(np.float32, copy=False)
    arr = arr - np.array(DNN_MEAN, dtype=np.float32)  # subtract per-channel mean
    arr = arr * DNN_SCALE
    # NCHW float32
    blob = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(blob)


def boxes_from_dnn_output(out: np.ndarray,
                          frame_w: int, frame_h: int,
                          conf_thresh: float) -> list[Box]:
    """Decode the standard Res10-SSD output ``[1, 1, N, 7]`` into boxes."""
    boxes: list[Box] = []
    for i in range(out.shape[2]):
        conf = float(out[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = (out[0, 0, i, 3:7] * np.array([frame_w, frame_h,
                                                        frame_w, frame_h])).astype("int")
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame_w, x2); y2 = min(frame_h, y2)
        w = x2 - x1; h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        boxes.append((int(x1), int(y1), int(w), int(h), conf))
    return boxes


# ---------- CPU baseline ----------

class CPUBackend:
    name = "cpu"
    vendor = "host"

    def __init__(self, proto: Path = CAFFE_PROTO_DEFAULT,
                 model: Path = CAFFE_MODEL_DEFAULT,
                 target_size: int = DNN_INPUT_SIZE):
        if not proto.exists() or not model.exists():
            raise FileNotFoundError(
                f"CPU backend needs {proto} and {model}. "
                "Download from https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector."
            )
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.target_size = target_size
        self._info = BackendInfo(
            name=self.name, vendor=self.vendor,
            device="cpu", detail=f"cv2.dnn Caffe target_size={target_size}"
        )

    def info(self) -> BackendInfo:
        return self._info

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.5) -> list[Box]:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (self.target_size, self.target_size)),
            DNN_SCALE, (self.target_size, self.target_size),
            DNN_MEAN,
        )
        self.net.setInput(blob)
        out = self.net.forward()
        return boxes_from_dnn_output(out, w, h, conf_thresh)


# ---------- cv2 OpenCL / Vulkan GPU targets ----------

class OpenCLBackend:
    """cv2.dnn with DNN_TARGET_OPENCL — wraps Apple's Metal via OpenCL on macOS,
    and Intel/AMD/NVIDIA drivers on Linux/Windows. No model conversion needed:
    it loads the same Caffe weights as the CPU baseline."""
    name = "cv2_opencl"
    vendor = "opencl"

    def __init__(self, proto: Path = CAFFE_PROTO_DEFAULT,
                 model: Path = CAFFE_MODEL_DEFAULT,
                 target_size: int = DNN_INPUT_SIZE):
        if not proto.exists() or not model.exists():
            raise FileNotFoundError(
                f"OpenCL backend needs {proto} and {model}."
            )
        try:
            self.net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        except cv2.error as e:
            raise RuntimeError(f"OpenCL target unavailable: {e}") from e
        self.target_size = target_size
        self._info = BackendInfo(
            name=self.name, vendor=self.vendor, device="opencl-gpu",
            detail=f"cv2.dnn OpenCL target_size={target_size}"
        )

    def info(self) -> BackendInfo:
        return self._info

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.5) -> list[Box]:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (self.target_size, self.target_size)),
            DNN_SCALE, (self.target_size, self.target_size),
            DNN_MEAN,
        )
        self.net.setInput(blob)
        out = self.net.forward()
        return boxes_from_dnn_output(out, w, h, conf_thresh)


class VulkanBackend:
    """cv2.dnn with DNN_TARGET_VULKAN — same model, alternative low-overhead
    GPU path on AMD/NVIDIA where Vulkan ICDs are present."""
    name = "cv2_vulkan"
    vendor = "vulkan"

    def __init__(self, proto: Path = CAFFE_PROTO_DEFAULT,
                 model: Path = CAFFE_MODEL_DEFAULT,
                 target_size: int = DNN_INPUT_SIZE):
        if not proto.exists() or not model.exists():
            raise FileNotFoundError(
                f"Vulkan backend needs {proto} and {model}."
            )
        try:
            self.net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_VULKAN)
        except cv2.error as e:
            raise RuntimeError(f"Vulkan target unavailable: {e}") from e
        self.target_size = target_size
        self._info = BackendInfo(
            name=self.name, vendor=self.vendor, device="vulkan-gpu",
            detail=f"cv2.dnn Vulkan target_size={target_size}"
        )

    def info(self) -> BackendInfo:
        return self._info

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.5) -> list[Box]:
        return OpenCLBackend.detect(self, frame_bgr, conf_thresh)


# ---------- onnxruntime-based backends ----------

class _OrtBackend:
    """Common scaffolding for an onnxruntime execution provider.

    Subclasses set ``ep_name`` (the ORT execution-provider name) and
    ``provider_options`` (per-vendor config). The constructor attempts to
    instantiate the EP; if it raises, ``available()`` returns False.
    """

    ep_name: str = ""
    ep_label: str = ""
    vendor: str = ""

    def __init__(self, onnx_path: Path = ONNX_MODEL_DEFAULT,
                 intra_op_threads: int = 1,
                 provider_options: Optional[dict] = None):
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise RuntimeError("onnxruntime is required for this backend") from e
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{self.ep_label} backend needs ONNX model at {onnx_path}. "
                f"Run tools/convert_res10_to_onnx.py to create it."
            )
        providers = [self.ep_name, "CPUExecutionProvider"]  # fall back to CPU
        opts = [provider_options or {}, {}]
        try:
            self.sess = ort.InferenceSession(
                str(onnx_path),
                providers=list(zip(providers, opts)),
            )
        except Exception as e:
            raise RuntimeError(
                f"{self.ep_label}: failed to create InferenceSession: {e}"
            ) from e
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        # Confirm we actually got the requested EP (otherwise ORT silently
        # falls back to CPU; we want to surface that).
        active = self.sess.get_providers()
        self._gpu_active = active[0] == self.ep_name
        self._info = BackendInfo(
            name=self.ep_label.lower().replace(" ", "_"),
            vendor=self.vendor,
            device=active[0],
            detail=f"onnxruntime providers={active} intra_op={intra_op_threads}",
        )

    def info(self) -> BackendInfo:
        return self._info

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.5) -> list[Box]:
        h, w = frame_bgr.shape[:2]
        blob = blob_from_bgr(frame_bgr)
        out = self.sess.run([self.output_name], {self.input_name: blob})[0]
        return boxes_from_dnn_output(out, w, h, conf_thresh)


class CoreMLBackend(_OrtBackend):
    """Apple Silicon / Intel Mac — uses CoreML EP, which dispatches to ANE
    and the integrated GPU."""
    ep_name = "CoreMLExecutionProvider"
    ep_label = "CoreML"
    vendor = "Apple"

    def __init__(self, onnx_path: Path = ONNX_MODEL_DEFAULT):
        # CoreML EP accepts ``ModelFormat``/``MLComputeUnits`` etc.; we leave
        # the defaults so ORT picks ANE when available, GPU otherwise.
        super().__init__(onnx_path, provider_options=None)


class CUDABackend(_OrtBackend):
    """NVIDIA — uses CUDA EP (cuDNN)."""
    ep_name = "CUDAExecutionProvider"
    ep_label = "CUDA"
    vendor = "NVIDIA"

    def __init__(self, onnx_path: Path = ONNX_MODEL_DEFAULT, device_id: int = 0):
        opts = {"device_id": device_id}
        super().__init__(onnx_path, provider_options=opts)


class ROCmBackend(_OrtBackend):
    """AMD ROCm on Linux — uses ROCm EP (mirrors CUDA API)."""
    ep_name = "ROCMExecutionProvider"
    ep_label = "ROCm"
    vendor = "AMD"


class DirectMLBackend(_OrtBackend):
    """AMD/NVIDIA on Windows — uses DirectML EP."""
    ep_name = "DmlExecutionProvider"
    ep_label = "DirectML"
    vendor = "AMD/NVIDIA(Win)"


class ACLBackend(_OrtBackend):
    """Huawei Ascend — uses ACL EP (CANN runtime)."""
    ep_name = "AclExecutionProvider"
    ep_label = "AscendACL"
    vendor = "Huawei Ascend"


class MLUBackend(_OrtBackend):
    """Cambricon MLU — uses Cambricon EP."""
    ep_name = "CambriconExecutionProvider"
    ep_label = "CambriconMLU"
    vendor = "Cambricon"


# ---------- AutoBackend dispatcher ----------

_BACKEND_REGISTRY: list[type] = [
    # Order matters: prefer Metal-via-OpenCL on Mac (no conversion needed,
    # bit-exact with CPU), then CoreML/ONNX, then cross-platform cv2 vulkan.
    OpenCLBackend,
    VulkanBackend,
    CoreMLBackend,
    CUDABackend,
    ROCmBackend,
    DirectMLBackend,
    ACLBackend,
    MLUBackend,
]


def available_backends() -> list[tuple[str, type]]:
    """Probe each backend; return [(label, class)] for those that initialise
    on this host."""
    out: list[tuple[str, type]] = []
    for cls in _BACKEND_REGISTRY:
        try:
            cls()
            out.append((cls.ep_label if hasattr(cls, "ep_label") else cls.name, cls))
        except Exception:
            continue
    return out


def auto_pick() -> DetectorBackend:
    """Return the first backend that successfully probes on this host."""
    for cls in _BACKEND_REGISTRY:
        try:
            return cls()
        except Exception as e:
            print(f"[auto] skip {getattr(cls, 'ep_label', cls.__name__)}: {e}",
                  file=sys.stderr)
    raise RuntimeError("No GPU backend available on this host")


def make_backend(name: str) -> DetectorBackend:
    name = name.lower().replace("-", "_")
    table = {
        "cpu": CPUBackend,
        "opencl": OpenCLBackend,
        "cv2_opencl": OpenCLBackend,
        "vulkan": VulkanBackend,
        "cv2_vulkan": VulkanBackend,
        "coreml": CoreMLBackend,
        "cuda": CUDABackend,
        "rocm": ROCmBackend,
        "directml": DirectMLBackend,
        "acl": ACLBackend,
        "ascend": ACLBackend,
        "mlu": MLUBackend,
        "cambricon": MLUBackend,
    }
    cls = table.get(name)
    if cls is None:
        raise ValueError(f"Unknown backend {name!r}; choices: {sorted(table)}")
    return cls()


if __name__ == "__main__":
    # Probe script: print which backends probe successfully on this host.
    print("== probing GPU backends ==")
    found = available_backends()
    if not found:
        print("no GPU backend available (only CPU works)")
    for label, cls in found:
        b = cls()
        info = b.info()
        print(f"  [OK] {label:<14} vendor={info.vendor:<14} device={info.device}")
    print(f"== {len(found)} backend(s) available ==")