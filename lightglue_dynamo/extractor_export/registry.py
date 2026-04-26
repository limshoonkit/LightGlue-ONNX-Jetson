"""Extractor metadata for notebooks and tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ExtractorSpec:
    """Static contract for ONNX extractor (phase 1)."""

    id: str
    input_name: str
    input_channels: int
    spatial_divisor: int
    min_opset: int
    output_names: tuple[str, ...]
    notes: str
    preprocess: Callable[[np.ndarray], np.ndarray]


def _sp_pre(img: np.ndarray) -> np.ndarray:
    from lightglue_dynamo.preprocessors import SuperPointPreprocessor

    return SuperPointPreprocessor.preprocess(img).astype(np.float32)


def _sp_open_pre(img: np.ndarray) -> np.ndarray:
    from lightglue_dynamo.preprocessors import SuperPointOpenPreprocessor

    return SuperPointOpenPreprocessor.preprocess(img).astype(np.float32)


def _disk_pre(img: np.ndarray) -> np.ndarray:
    from lightglue_dynamo.preprocessors import DISKPreprocessor

    return DISKPreprocessor.preprocess(img).astype(np.float32)


def _aliked_pre(img: np.ndarray) -> np.ndarray:
    from lightglue_dynamo.preprocessors import ALIKEDPreprocessor

    return ALIKEDPreprocessor.preprocess(img).astype(np.float32)


def _rgb_nchw_01(img: np.ndarray) -> np.ndarray:
    """BGR uint8 (N,H,W,3) -> RGB NCHW float [0,1]."""
    x = img[..., ::-1].astype(np.float32) / 255.0
    if x.ndim == 3:
        x = x[np.newaxis, ...]
    return np.transpose(x, (0, 3, 1, 2))


def _xfeat_pre(img: np.ndarray) -> np.ndarray:
    """Match upstream: grayscale NCHW in [0,1], H/W multiple of 32 via resize in torch at inference."""
    if img.ndim == 3:
        img = img[np.newaxis, ...]
    gray = (img[..., ::-1].astype(np.float32) * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=-1)
    return gray[:, np.newaxis, :, :].astype(np.float32) / 255.0


EXTRACTOR_REGISTRY: dict[str, ExtractorSpec] = {
    "superpoint": ExtractorSpec(
        id="superpoint",
        input_name="images",
        input_channels=1,
        spatial_divisor=8,
        min_opset=17,
        output_names=("keypoints", "keypoint_scores", "descriptors"),
        notes="CVG SuperPoint; weights from torch hub inside SuperPoint().",
        preprocess=_sp_pre,
    ),
    "superpoint_open": ExtractorSpec(
        id="superpoint_open",
        input_name="images",
        input_channels=1,
        spatial_divisor=8,
        min_opset=17,
        output_names=("keypoints", "keypoint_scores", "descriptors", "num_keypoints"),
        notes="TensorFlow-port weights; preprocessor pads H,W to multiple of 8.",
        preprocess=_sp_open_pre,
    ),
    "disk": ExtractorSpec(
        id="disk",
        input_name="images",
        input_channels=3,
        spatial_divisor=16,
        min_opset=17,
        output_names=("keypoints", "keypoint_scores", "descriptors"),
        notes="DISK; hub weights inside DISK().",
        preprocess=_disk_pre,
    ),
    "aliked_n16": ExtractorSpec(
        id="aliked_n16",
        input_name="images",
        input_channels=3,
        spatial_divisor=32,
        min_opset=19,
        output_names=("keypoints", "keypoint_scores", "descriptors", "num_keypoints"),
        notes="Uses ONNX DeformConv (opset>=19). TensorRT support depends on TRT version.",
        preprocess=_aliked_pre,
    ),
    "aliked_n16rot": ExtractorSpec(
        id="aliked_n16rot",
        input_name="images",
        input_channels=3,
        spatial_divisor=32,
        min_opset=19,
        output_names=("keypoints", "keypoint_scores", "descriptors", "num_keypoints"),
        notes="Same as aliked_n16 with rotation-augmented training variant.",
        preprocess=_aliked_pre,
    ),
    "aliked_n32": ExtractorSpec(
        id="aliked_n32",
        input_name="images",
        input_channels=3,
        spatial_divisor=32,
        min_opset=19,
        output_names=("keypoints", "keypoint_scores", "descriptors", "num_keypoints"),
        notes="ALIKED-n32 (M=32).",
        preprocess=_aliked_pre,
    ),
    "aliked_t16": ExtractorSpec(
        id="aliked_t16",
        input_name="images",
        input_channels=3,
        spatial_divisor=32,
        min_opset=19,
        output_names=("keypoints", "keypoint_scores", "descriptors", "num_keypoints"),
        notes="ALIKED-t16 (smaller).",
        preprocess=_aliked_pre,
    ),
    "raco": ExtractorSpec(
        id="raco",
        input_name="images",
        input_channels=3,
        spatial_divisor=32,
        min_opset=17,
        output_names=("keypoints", "keypoint_scores"),
        notes="ref/RaCo: no descriptor map in checkpoint; matcher raco_aliked_lightglue may pair with a separate descriptor network.",
        preprocess=_rgb_nchw_01,
    ),
    "xfeat": ExtractorSpec(
        id="xfeat",
        input_name="images",
        input_channels=1,
        spatial_divisor=32,
        min_opset=17,
        output_names=("keypoints", "keypoint_scores", "descriptors", "num_keypoints"),
        notes=(
            "Padded wrapper around ref XFeat; PyTorch legacy ONNX may fail on Unfold/InstanceNorm in XFeatModel—"
            "try fixed H,W, newer torch/onnx, or a slim export-only backbone."
        ),
        preprocess=_xfeat_pre,
    ),
}


def build_dynamic_axes(
    *,
    input_name: str,
    output_names: tuple[str, ...],
    dynamic_batch: bool,
    dynamic_hw: bool,
) -> dict[str, dict[int, str]]:
    """Build ``dynamic_axes`` for ``torch.onnx.export`` (dim 0 = batch, 2/3 = H/W on NCHW images)."""
    axes: dict[str, dict[int, str]] = {input_name: {}}
    if dynamic_batch:
        axes[input_name][0] = "batch_size"
    if dynamic_hw:
        axes[input_name][2] = "height"
        axes[input_name][3] = "width"
    for n in output_names:
        axes[n] = {}
        if dynamic_batch:
            axes[n][0] = "batch_size"
    return axes


def get_extractor_spec(extractor_id: str) -> ExtractorSpec:
    key = extractor_id.lower().strip()
    if key not in EXTRACTOR_REGISTRY:
        raise KeyError(f"Unknown extractor {extractor_id!r}; known: {sorted(EXTRACTOR_REGISTRY)}")
    return EXTRACTOR_REGISTRY[key]
