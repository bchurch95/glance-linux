"""ArcFace embedding via ONNX Runtime.

The model is the InsightFace ArcFace ONNX, used directly. Upstream converts it
to Core ML (`tools/convert_arcface.py`); on Linux that conversion step simply
does not exist — the original ONNX is what runs. Embeddings are therefore
numerically comparable with the macOS app's, which matters if you ever want to
check an enrollment against both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

EMBEDDING_DIMENSIONS = 512


class ArcFaceEmbedder:
    """Turns an aligned 112x112 RGB crop into a normalized 512-d embedding."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        providers: Optional[list[str]] = None,
        device: str = "auto",
    ) -> None:
        from . import paths

        self.model_path = Path(model_path or paths.arcface_model())
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {self.model_path}. "
                "Fetch it with `glancectl fetch-model`."
            )

        self._compiled = None
        self._output = None
        self._ort_session = None
        self.device = "CPU"

        # Try hardware NPU acceleration via OpenVINO (e.g. Intel Lunar Lake NPU 4)
        if device in ("auto", "npu", "NPU") and providers is None:
            try:
                import openvino as ov

                core = ov.Core()
                if "NPU" in core.available_devices:
                    model = core.read_model(str(self.model_path))
                    # Reshape to static batch size 1 required by NPU compiler
                    model.reshape([1, 3, 112, 112])
                    self._compiled = core.compile_model(model, "NPU")
                    self._output = self._compiled.output(0)
                    self.device = "NPU"
            except Exception:
                pass

        if self._compiled is None:
            import onnxruntime

            self._ort_session = onnxruntime.InferenceSession(
                str(self.model_path), providers=providers or ["CPUExecutionProvider"]
            )
            self._ort_input = self._ort_session.get_inputs()[0].name

    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        """Embed one aligned crop. Returns an L2-normalized (512,) float32 array.

        Normalizing here rather than at each comparison site means cosine
        similarity is a plain dot product everywhere downstream, and no caller
        can forget to normalize and silently get a distance that drifts with
        image brightness.
        """
        if aligned_rgb.shape[:2] != (112, 112):
            raise ValueError(f"expected a 112x112 crop, got {aligned_rgb.shape[:2]}")
        # ArcFace's standard input scaling: [0, 255] -> [-1, 1], NCHW.
        blob = (aligned_rgb.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        if self._compiled is not None:
            raw = self._compiled([blob])[self._output][0]
        else:
            raw = self._ort_session.run(None, {self._ort_input: blob})[0][0]
        norm = float(np.linalg.norm(raw))
        if norm <= 0:
            raise ValueError("embedder returned a zero vector")
        return (raw / norm).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized embeddings, in [-1, 1]."""
    return float(np.dot(a, b))
