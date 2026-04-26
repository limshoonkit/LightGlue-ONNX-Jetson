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
    opset: int = 17,
    device: str = "cpu",
) -> Path:
    """Export a LightGlue / LighterGlue matcher to ONNX.

    The exported model has fixed keypoint count ``K = num_keypoints`` and a dynamic
    match-count dimension on the outputs.

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
    """
    model = load_lightglue_local(
        weights_path,
        input_dim=input_dim,
        descriptor_dim=descriptor_dim,
        num_heads=num_heads,
        n_layers=n_layers,
        filter_threshold=filter_threshold,
        state_dict_prefix=state_dict_prefix,
    ).to(device)

    K, D = num_keypoints, input_dim
    kpts0 = torch.rand(1, K, 2, device=device) * 2 - 1
    kpts1 = torch.rand(1, K, 2, device=device) * 2 - 1
    desc0 = torch.randn(1, K, D, device=device)
    desc1 = torch.randn(1, K, D, device=device)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # dynamo=False forces the TorchScript-based exporter (default changed to True in PT 2.9+).
    torch.onnx.export(
        model,
        (kpts0, kpts1, desc0, desc1),
        str(out),
        input_names=["kpts0", "kpts1", "desc0", "desc1"],
        output_names=["matches0", "mscores0"],
        opset_version=opset,
        dynamic_axes={"matches0": {0: "num_matches"}, "mscores0": {0: "num_matches"}},
        dynamo=False,
    )

    return out
