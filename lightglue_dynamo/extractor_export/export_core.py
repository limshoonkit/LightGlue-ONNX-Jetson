"""Unified ONNX export for local feature extractors (extractor-only, phase 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from lightglue_dynamo.models import ALIKED, DISK, SuperPoint, SuperPointOpen

from .adapter_raco import RaCoOnnxWrapper, load_raco
from .adapter_xfeat import XFeatPaddedOnnxWrapper, load_xfeat
from .symbolic import register_deform_conv2d_onnx
from .validation import validate_onnx
from .wrappers import (
    AlikedOnnxWrapper,
    DiskOnnxWrapper,
    SuperPointCvgOnnxWrapper,
    SuperPointOpenOnnxWrapper,
    configure_aliked_for_export,
)

_ALIKED_NAMES = {
    "aliked_n16": "aliked-n16",
    "aliked_n16rot": "aliked-n16rot",
    "aliked_n32": "aliked-n32",
    "aliked_t16": "aliked-t16",
}


def _trace_batch(batch_size: int) -> int:
    return batch_size if batch_size > 0 else 2


def _build_dynamic_axes_images_only(
    *,
    dynamic_batch: bool,
    dynamic_hw: bool,
) -> dict[str, dict[int, str]]:
    d: dict[str, dict[int, str]] = {"images": {}}
    if dynamic_batch:
        d["images"][0] = "batch_size"
    if dynamic_hw:
        d["images"][2] = "height"
        d["images"][3] = "width"
    return d


def _add_output_batch_axis(dynamic_axes: dict[str, dict[int, str]], names: list[str], *, dynamic_batch: bool) -> None:
    if not dynamic_batch:
        return
    for n in names:
        dynamic_axes.setdefault(n, {})
        dynamic_axes[n][0] = "batch_size"


def export_extractor_onnx(
    extractor_id: str,
    output_path: str | Path,
    *,
    weights_root: str | Path,
    batch_size: int = 2,
    height: int = 384,
    width: int = 640,
    max_keypoints: int = 256,
    opset: int = 17,
    dynamic_batch: bool = False,
    dynamic_hw: bool = False,
    device: str = "cpu",
    superpoint_open_checkpoint: str | Path | None = None,
    aliked_checkpoint: str | Path | None = None,
    raco_checkpoint: str | Path | None = None,
    xfeat_checkpoint: str | Path | None = None,
) -> None:
    """
    Export a single extractor to ONNX.

    Parameters
    ----------
    extractor_id:
        One of: ``superpoint``, ``superpoint_open``, ``disk``, ``aliked_n16``, ``aliked_n16rot``,
        ``aliked_n32``, ``aliked_t16``, ``xfeat``, ``raco``.
    weights_root:
        Directory containing default weight filenames (e.g. ``superpoint_v6_from_tf.pth``).
    batch_size:
        Trace size; use ``0`` with ``dynamic_batch=True`` to mark batch axis as dynamic.
    superpoint_open_checkpoint / aliked_checkpoint / raco_checkpoint / xfeat_checkpoint:
        Optional explicit paths; otherwise resolved under ``weights_root`` with conventional names.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = Path(weights_root)
    dev = torch.device(device)
    eid = extractor_id.lower().strip()
    b = _trace_batch(batch_size)

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive for extractor tracing")

    if eid == "superpoint":
        model = SuperPoint(num_keypoints=max_keypoints).eval().to(dev)
        wrapped = SuperPointCvgOnnxWrapper(model).eval().to(dev)
        dummy = torch.randn(b, 1, height, width, device=dev)
        outs = ["keypoints", "keypoint_scores", "descriptors"]
        dynamic_axes = _build_dynamic_axes_images_only(dynamic_batch=dynamic_batch, dynamic_hw=dynamic_hw)
        for n in outs:
            dynamic_axes[n] = {}
        _add_output_batch_axis(dynamic_axes, outs, dynamic_batch=dynamic_batch)
        torch.onnx.export(
            wrapped,
            dummy,
            str(out),
            input_names=["images"],
            output_names=outs,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
        validate_onnx(out)
        return

    if eid == "disk":
        model = DISK(num_keypoints=max_keypoints).eval().to(dev)
        wrapped = DiskOnnxWrapper(model).eval().to(dev)
        dummy = torch.randn(b, 3, height, width, device=dev)
        outs = ["keypoints", "keypoint_scores", "descriptors"]
        dynamic_axes = _build_dynamic_axes_images_only(dynamic_batch=dynamic_batch, dynamic_hw=dynamic_hw)
        for n in outs:
            dynamic_axes[n] = {}
        _add_output_batch_axis(dynamic_axes, outs, dynamic_batch=dynamic_batch)
        torch.onnx.export(
            wrapped,
            dummy,
            str(out),
            input_names=["images"],
            output_names=outs,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
        validate_onnx(out)
        return

    if eid == "superpoint_open":
        ck = Path(superpoint_open_checkpoint) if superpoint_open_checkpoint else wr / "superpoint_v6_from_tf.pth"
        model = SuperPointOpen(detection_threshold=0.005, nms_radius=5, max_num_keypoints=max_keypoints).eval()
        sd = torch.load(str(ck), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        model = model.to(dev)
        for p in model.parameters():
            p.requires_grad = False
        wrapped = SuperPointOpenOnnxWrapper(model).eval().to(dev)
        dummy = torch.randn(b, 1, height, width, device=dev)
        outs = ["keypoints", "keypoint_scores", "descriptors", "num_keypoints"]
        dynamic_axes = {n: {} for n in ["images", *outs]}
        if dynamic_batch:
            for n in ["images", *outs]:
                dynamic_axes[n][0] = "batch_size"
        torch.onnx.export(
            wrapped,
            dummy,
            str(out),
            input_names=["images"],
            output_names=outs,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            dynamo=False,
        )
        validate_onnx(out)
        return

    if eid in _ALIKED_NAMES:
        register_deform_conv2d_onnx(19)
        model_name = _ALIKED_NAMES[eid]
        ck = Path(aliked_checkpoint) if aliked_checkpoint else wr / f"{model_name}.pth"
        extractor = ALIKED(
            model_name=model_name,
            device="cpu",
            n_limit=max_keypoints,
            pretrained_path=str(ck),
        )
        extractor = configure_aliked_for_export(extractor, max_keypoints)
        extractor = extractor.eval().to(dev)
        wrapped = AlikedOnnxWrapper(extractor, max_kps=max_keypoints).eval().to(dev)
        dummy = torch.randn(b, 3, height, width, device=dev)
        outs = ["keypoints", "keypoint_scores", "descriptors", "num_keypoints"]
        dynamic_axes = {n: {} for n in ["images", *outs]}
        if dynamic_batch:
            for n in ["images", *outs]:
                dynamic_axes[n][0] = "batch_size"
        eff_opset = max(opset, 19)
        torch.onnx.export(
            wrapped,
            dummy,
            str(out),
            input_names=["images"],
            output_names=outs,
            opset_version=eff_opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
        validate_onnx(out)
        return

    if eid == "raco":
        ck = Path(raco_checkpoint) if raco_checkpoint else wr / "raco.pth"
        raco = load_raco(ck, max_keypoints=max_keypoints).to(dev)
        wrapped = RaCoOnnxWrapper(raco).eval().to(dev)
        dummy = torch.randn(b, 3, height, width, device=dev)
        outs = ["keypoints", "keypoint_scores"]
        dynamic_axes = {n: {} for n in ["images", *outs]}
        if dynamic_batch:
            for n in ["images", *outs]:
                dynamic_axes[n][0] = "batch_size"
        if dynamic_hw:
            dynamic_axes["images"][2] = "height"
            dynamic_axes["images"][3] = "width"
        torch.onnx.export(
            wrapped,
            dummy,
            str(out),
            input_names=["images"],
            output_names=outs,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
        validate_onnx(out)
        return

    if eid == "xfeat":
        ck = Path(xfeat_checkpoint) if xfeat_checkpoint else wr / "xfeat.pt"
        xf = load_xfeat(ck, device=device)
        wrapped = XFeatPaddedOnnxWrapper(xf, max_k=max_keypoints).eval().to(dev)
        dummy = torch.randn(b, 1, height, width, device=dev)
        outs = ["keypoints", "keypoint_scores", "descriptors", "num_keypoints"]
        dynamic_axes = {n: {} for n in ["images", *outs]}
        if dynamic_batch:
            for n in ["images", *outs]:
                dynamic_axes[n][0] = "batch_size"
        torch.onnx.export(
            wrapped,
            dummy,
            str(out),
            input_names=["images"],
            output_names=outs,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
        validate_onnx(out)
        return

    raise ValueError(f"Unknown extractor_id: {extractor_id!r}")


def export_aliked_dense_extractor_onnx(
    output_path: str | Path,
    *,
    aliked_checkpoint: str | Path,
    model_name: str = "aliked-n16",
    batch_size: int = 2,
    height: int = 480,
    width: int = 768,
    opset: int = 19,
    device: str = "cpu",
) -> None:
    """Export ALIKED dense headless extractor (backbone + score_head only) to ONNX.

    Inputs:  image (B, 3, H, W)  float32, RGB in [0, 1]
    Output:  score_map   (B, 1, H, W)    float32
             feature_map (B, dim, H, W)  float32, L2-normalised
    """
    from .wrappers import AlikedDenseExtractorWrapper

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    eff_opset = max(opset, 19)

    register_deform_conv2d_onnx(eff_opset)

    if model_name not in _ALIKED_NAMES.values():
        raise ValueError(f"Unknown ALIKED model_name {model_name!r}; expected one of {list(_ALIKED_NAMES.values())}")

    model = ALIKED(
        model_name=model_name,
        device=str(dev),
        pretrained_path=str(Path(aliked_checkpoint)),
    ).eval().to(dev)

    wrapped = AlikedDenseExtractorWrapper(model).eval().to(dev)

    dummy_image = torch.randn(batch_size, 3, height, width, device=dev)

    torch.onnx.export(
        wrapped,
        (dummy_image,),
        str(out),
        input_names=["image"],
        output_names=["score_map", "feature_map"],
        opset_version=eff_opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    validate_onnx(out)


def export_aliked_desc_head_only_onnx(
    output_path: str | Path,
    *,
    aliked_checkpoint: str | Path,
    model_name: str = "aliked-n16",
    batch_size: int = 2,
    height: int = 480,
    width: int = 768,
    max_keypoints: int = 256,
    feature_dim: int = 128,
    opset: int = 19,
    device: str = "cpu",
) -> None:
    """Export ALIKED desc-head-only ONNX (no backbone, vectorised SDDH).

    Used as Engine B in the three-stage pipeline. Pairs with the dense headless
    extractor whose feature_map output feeds directly into this engine, eliminating
    the redundant backbone pass that the legacy ``aliked_n16_describe`` engine
    incurs.

    Inputs:  feature_map (B, feature_dim, H, W)  float32
             kpts_n      (B, K, 2)               float32, in [-1, 1] (x, y) order
    Output:  descriptors (B, K, feature_dim)     float32, L2-normalised
    """
    from .wrappers import AlikedDescHeadOnlyWrapper

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    eff_opset = max(opset, 19)

    register_deform_conv2d_onnx(eff_opset)

    if model_name not in _ALIKED_NAMES.values():
        raise ValueError(f"Unknown ALIKED model_name {model_name!r}; expected one of {list(_ALIKED_NAMES.values())}")

    model = ALIKED(
        model_name=model_name,
        device=str(dev),
        pretrained_path=str(Path(aliked_checkpoint)),
    ).eval().to(dev)

    wrapped = AlikedDescHeadOnlyWrapper(model).eval().to(dev)

    dummy_feature_map = torch.randn(batch_size, feature_dim, height, width, device=dev)
    dummy_kpts_n = torch.empty(batch_size, max_keypoints, 2, device=dev).uniform_(-0.9, 0.9)

    torch.onnx.export(
        wrapped,
        (dummy_feature_map, dummy_kpts_n),
        str(out),
        input_names=["feature_map", "kpts_n"],
        output_names=["descriptors"],
        opset_version=eff_opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    validate_onnx(out)


def export_aliked_describe_onnx(
    output_path: str | Path,
    *,
    aliked_checkpoint: str | Path,
    batch_size: int = 2,
    height: int = 480,
    width: int = 768,
    max_keypoints: int = 256,
    opset: int = 19,
    device: str = "cpu",
) -> None:
    """Export ALIKED-n16 describe-at-external-keypoints to ONNX.

    Inputs:  image  (B, 3, H, W)  float32, RGB in [0, 1]
             kpts_n (B, K, 2)     keypoints anisotropically normalised to [-1, 1]
    Output:  descriptors (B, K, 128)
    """
    from .wrappers import AlikedDescribeWrapper

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    b = batch_size
    eff_opset = max(opset, 19)

    register_deform_conv2d_onnx(eff_opset)

    model = ALIKED(
        model_name="aliked-n16",
        device=str(dev),
        pretrained_path=str(Path(aliked_checkpoint)),
    ).eval().to(dev)

    wrapped = AlikedDescribeWrapper(model).eval().to(dev)

    dummy_image = torch.randn(b, 3, height, width, device=dev)
    dummy_kpts = torch.zeros(b, max_keypoints, 2, device=dev)

    torch.onnx.export(
        wrapped,
        (dummy_image, dummy_kpts),
        str(out),
        input_names=["image", "kpts_n"],
        output_names=["descriptors"],
        opset_version=eff_opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    validate_onnx(out)


def list_supported_extractors() -> list[str]:
    return [
        "superpoint",
        "superpoint_open",
        "disk",
        *_ALIKED_NAMES.keys(),
        "raco",
        "xfeat",
    ]
