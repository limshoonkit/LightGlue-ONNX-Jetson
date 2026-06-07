"""Matcher ONNX export for LightGlue / LighterGlue.

Wraps the project-native ``lightglue_dynamo.models.lightglue.LightGlue`` (which uses a
concatenated-pair batch format) in a thin adapter that presents the 4-input interface
required for clean ONNX tracing:

    forward(kpts0, kpts1, desc0, desc1) -> (matches0, mscores0)

Weight loading bypasses ``torch.hub`` so local ``.pth`` / ``.pt`` checkpoints can be used
directly, including joint checkpoints (e.g. xfeat-lighterglue.pt) that pack extractor and
matcher weights under separate key prefixes.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_hub_loader(state_dict: dict):
    """Redirect ``torch.hub.load_state_dict_from_url`` to a preloaded dict."""
    import torch.hub as _hub
    _orig = _hub.load_state_dict_from_url
    _hub.load_state_dict_from_url = lambda *a, **kw: state_dict
    try:
        yield
    finally:
        _hub.load_state_dict_from_url = _orig


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class LightGlueExporter(nn.Module):
    """4-input ONNX export adapter around the dynamo LightGlue.

    The dynamo model expects concatenated-pair inputs ``(2, N, *)``; this wrapper
    accepts the four separate tensors that make up one matching pair and unpacks the
    ``(M, 3)`` match output (batch_idx, kpt_idx0, kpt_idx1) into ``(M, 2)``.

    I/O contract::

        Inputs:
          kpts0  (1, K, 2)  isotropic-normalised keypoints, image 0
          kpts1  (1, K, 2)  isotropic-normalised keypoints, image 1
          desc0  (1, K, D)  descriptors, image 0  (D = input_dim)
          desc1  (1, K, D)  descriptors, image 1

        Outputs:
          matches0  (M, 2)  per-match [idx_in_kpts0, idx_in_kpts1]
          mscores0  (M,)    match confidence scores
    """

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kpts  = torch.cat([kpts0, kpts1], 0)   # (2, K, 2)
        descs = torch.cat([desc0, desc1], 0)   # (2, K, D)
        matches, mscores = self.core(kpts, descs)  # (M, 3), (M,)
        # Drop the batch-index column (always 0 for a single pair).
        return matches[:, 1:], mscores              # (M, 2), (M,)


class LightGlueBatchedExporter(nn.Module):
    """Batched 4-input ONNX export adapter around the dynamo LightGlue.

    Identical to :class:`LightGlueExporter` but accepts a leading batch axis so a
    single inference can match ``B`` independent pairs at once.  The dynamo core
    uses a ``(2B, N, *)`` *interleaved* layout — rows ``2i`` and ``2i+1`` form pair
    ``i`` — so this wrapper interleaves the four per-side tensors into that layout
    and returns the raw ``(M, 3)`` match output so the caller can demultiplex
    matches by their batch index.

    I/O contract::

        Inputs:
          kpts0  (B, K, 2)  isotropic-normalised keypoints, image 0 of each pair
          kpts1  (B, K, 2)  isotropic-normalised keypoints, image 1 of each pair
          desc0  (B, K, D)  descriptors, image 0 of each pair
          desc1  (B, K, D)  descriptors, image 1 of each pair

        Outputs:
          matches0  (M, 3)  per-match [batch_idx, idx_in_kpts0, idx_in_kpts1]
          mscores0  (M,)    match confidence scores
    """

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Interleave to the dynamo (2B, N, *) layout: pair i -> rows 2i, 2i+1.
        kpts  = torch.stack([kpts0, kpts1], dim=1).flatten(0, 1)   # (2B, K, 2)
        descs = torch.stack([desc0, desc1], dim=1).flatten(0, 1)   # (2B, K, D)
        matches, mscores = self.core(kpts, descs)  # (M, 3), (M,)
        # Keep the batch-index column so callers can split matches per pair.
        return matches, mscores                     # (M, 3), (M,)


# Per-matcher architecture parameters. Weight filenames are resolved relative to
# a caller-provided weights directory. Mirrors the registry in
# notebooks/3-export_trt_matchers.ipynb so the CLI and notebook stay in sync.
MATCHER_REGISTRY: dict[str, dict] = {
    "superpoint": dict(weights="superpoint_lightglue.pth",
                       input_dim=256, descriptor_dim=256, n_layers=9, num_heads=4,
                       state_dict_prefix=None),
    "aliked":     dict(weights="aliked_lightglue.pth",
                       input_dim=128, descriptor_dim=256, n_layers=9, num_heads=4,
                       state_dict_prefix=None),
    # Architecturally identical to aliked_lightglue: RaCo keypoints + ALIKED descriptors.
    "raco":       dict(weights="raco_aliked_lightglue.pth",
                       input_dim=128, descriptor_dim=256, n_layers=9, num_heads=4,
                       state_dict_prefix=None),
    # OpenCV SIFT (128-d); posenc uses (x, y, scale, ori). No extractor ONNX export.
    "sift":       dict(weights="sift_lightglue.pth",
                       input_dim=128, descriptor_dim=256, n_layers=9, num_heads=4,
                       keypoint_dim=4, add_scale_ori=True,
                       state_dict_prefix=None),
    # LighterGlue joint checkpoint: matcher weights live under the "matcher." prefix.
    "xfeat":      dict(weights="xfeat-lighterglue.pt",
                       input_dim=64, descriptor_dim=96, n_layers=6, num_heads=1,
                       state_dict_prefix="matcher."),
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_lightglue_local(
    weights_path: str | Path,
    *,
    input_dim: int = 256,
    descriptor_dim: int = 256,
    num_heads: int = 4,
    n_layers: int = 9,
    filter_threshold: float = 0.1,
    state_dict_prefix: str | None = None,
    add_scale_ori: bool = False,
) -> LightGlueExporter:
    """Load LightGlue / LighterGlue from a local checkpoint and wrap for ONNX export.

    Parameters
    ----------
    weights_path:
        Local ``.pth`` / ``.pt`` file containing the matcher state dict.
    input_dim:
        Descriptor dimensionality produced by the paired feature extractor
        (e.g. 256 for SuperPoint, 128 for ALIKED, 64 for xFeat).
    descriptor_dim:
        Internal embedding dimension of the matcher (default 256; 96 for LighterGlue).
    num_heads:
        Number of attention heads (default 4; 1 for LighterGlue).
    n_layers:
        Number of transformer layers (default 9; 6 for LighterGlue).
    filter_threshold:
        Mutual-NN match confidence threshold.
    state_dict_prefix:
        Strip this prefix from all checkpoint keys before loading.  Use ``"matcher."``
        for joint checkpoints such as ``xfeat-lighterglue.pt``.
    add_scale_ori:
        When True, build the 4-input positional encoder used by SIFT / DoGHardNet
        checkpoints (``posenc.Wr`` expects 4-D keypoints: xy + scale + orientation).
    """
    from lightglue_dynamo.models.lightglue import LightGlue

    sd = torch.load(str(weights_path), map_location="cpu")
    if state_dict_prefix is not None:
        p = state_dict_prefix
        sd = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}

    with _patch_hub_loader(sd):
        core = LightGlue(
            url="local",
            input_dim=input_dim,
            descriptor_dim=descriptor_dim,
            num_heads=num_heads,
            n_layers=n_layers,
            filter_threshold=filter_threshold,
            add_scale_ori=add_scale_ori,
        )

    return LightGlueExporter(core).eval()


def export_matcher_onnx(
    weights_path: str | Path,
    output_path: str | Path,
    *,
    num_keypoints: int = 256,
    input_dim: int = 256,
    descriptor_dim: int = 256,
    num_heads: int = 4,
    n_layers: int = 9,
    filter_threshold: float = 0.1,
    state_dict_prefix: str | None = None,
    add_scale_ori: bool = False,
    keypoint_dim: int = 2,
    opset: int = 17,
    device: str = "cpu",
    batch_size: int = 1,
    dynamic_batch: bool = False,
) -> Path:
    """Export a LightGlue / LighterGlue matcher to ONNX.

    The exported model has fixed keypoint count ``K = num_keypoints`` and a dynamic
    match-count dimension on the outputs.

    By default a single-pair model is exported (``batch_size=1``): inputs are
    ``(1, K, *)`` and ``matches0`` is ``(M, 2)``.  With ``batch_size > 1`` or
    ``dynamic_batch=True`` the *batched* variant is exported instead: inputs are
    ``(B, K, *)`` and ``matches0`` is ``(M, 3)`` with a leading batch-index column,
    so one inference matches ``B`` independent pairs.

    Parameters
    ----------
    weights_path:
        Local checkpoint file.
    output_path:
        Destination ``.onnx`` file.
    num_keypoints:
        Fixed ``K`` for the exported graph (must match the extractor export).
    input_dim:
        Descriptor dimensionality (D) fed into the matcher.
    state_dict_prefix:
        Key prefix to strip (see :func:`load_lightglue_local`).
    opset:
        ONNX opset version (17 is sufficient for all supported matchers).
    batch_size:
        Number of pairs the exported graph matches per inference.  Used as the
        traced (and, unless ``dynamic_batch``, fixed) batch dimension.
    dynamic_batch:
        Mark the input batch axis dynamic so a single engine serves any batch
        size.  TensorRT still needs an optimisation profile (min/opt/max) at
        build time.  Implies the batched variant even when ``batch_size == 1``.
    """
    model = load_lightglue_local(
        weights_path,
        input_dim=input_dim,
        descriptor_dim=descriptor_dim,
        num_heads=num_heads,
        n_layers=n_layers,
        filter_threshold=filter_threshold,
        state_dict_prefix=state_dict_prefix,
        add_scale_ori=add_scale_ori,
    ).to(device)

    batched = batch_size > 1 or dynamic_batch
    if batched:
        # Rewrap the loaded dynamo core in the batched adapter.
        model = LightGlueBatchedExporter(model.core).eval().to(device)

    B = max(batch_size, 1)
    K, D = num_keypoints, input_dim
    kpt_dim = 4 if add_scale_ori else keypoint_dim
    kpts0 = torch.rand(B, K, kpt_dim, device=device) * 2 - 1
    kpts1 = torch.rand(B, K, kpt_dim, device=device) * 2 - 1
    desc0 = torch.randn(B, K, D, device=device)
    desc1 = torch.randn(B, K, D, device=device)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes: dict[str, dict[int, str]] = {
        "matches0": {0: "num_matches"},
        "mscores0": {0: "num_matches"},
    }
    if dynamic_batch:
        for name in ("kpts0", "kpts1", "desc0", "desc1"):
            dynamic_axes[name] = {0: "batch"}

    # dynamo=False forces the TorchScript-based exporter (default changed to True in PT 2.9+).
    torch.onnx.export(
        model,
        (kpts0, kpts1, desc0, desc1),
        str(out),
        input_names=["kpts0", "kpts1", "desc0", "desc1"],
        output_names=["matches0", "mscores0"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    return out


def export_matcher_from_config(
    cfg: dict,
    output_path: str | Path,
    *,
    num_keypoints: int = 256,
    opset: int = 17,
    device: str = "cpu",
    batch_size: int = 1,
    dynamic_batch: bool = False,
) -> Path:
    """Export a matcher using a notebook/CLI config dict.

    Expected keys: ``weights``, ``input_dim``, ``desc_dim`` (or ``descriptor_dim``),
    ``n_layers``, ``num_heads``, ``prefix`` (or ``state_dict_prefix``),
    optional ``add_scale_ori``, ``kpt_dim`` (or ``keypoint_dim``).
    """
    desc_dim = cfg.get("desc_dim", cfg.get("descriptor_dim", 256))
    prefix = cfg.get("prefix", cfg.get("state_dict_prefix"))
    kpt_dim = cfg.get("kpt_dim", cfg.get("keypoint_dim", 2))
    add_scale_ori = bool(cfg.get("add_scale_ori", kpt_dim >= 4))

    return export_matcher_onnx(
        cfg["weights"],
        output_path,
        num_keypoints=num_keypoints,
        input_dim=cfg["input_dim"],
        descriptor_dim=desc_dim,
        num_heads=cfg["num_heads"],
        n_layers=cfg["n_layers"],
        state_dict_prefix=prefix,
        add_scale_ori=add_scale_ori,
        keypoint_dim=kpt_dim,
        opset=opset,
        device=device,
        batch_size=batch_size,
        dynamic_batch=dynamic_batch,
    )
