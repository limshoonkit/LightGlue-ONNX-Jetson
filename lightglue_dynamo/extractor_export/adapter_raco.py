"""RaCo extractor ONNX adapter (keypoints + scores only; no descriptor head in ref model)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class RaCoOnnxWrapper(nn.Module):
    """
    Wraps RaCo ``forward({"image": ...})`` for ONNX.

    The public ``raco.pth`` checkpoint matches the score-head-only architecture
    (keypoints + keypoint_scores). Descriptor pairing for the raco_aliked_lightglue
    stack must use a separate ALIKED extractor.

    When ``covariance=True`` the RaCo covariance head is enabled and the wrapper emits a
    third output ``covariances`` (B, K, 2, 2) — the per-keypoint pixel covariance consumed by
    the DL-VINS reprojection-factor weighting (see .vscode/stereo_improvement design 4).
    """

    def __init__(self, raco_module: nn.Module, *, covariance: bool = False) -> None:
        super().__init__()
        self.raco = raco_module
        self.covariance = covariance

    def forward(self, images: torch.Tensor):
        out = self.raco.forward({"image": images})
        if self.covariance:
            return out["keypoints"], out["keypoint_scores"], out["covariances"]
        return out["keypoints"], out["keypoint_scores"]


def load_raco(weights_path: str | Path, *, max_keypoints: int = 2048,
              covariance: bool = False) -> nn.Module:
    from lightglue_dynamo.models.raco import RaCo

    model = RaCo(weights=str(weights_path), max_num_keypoints=max_keypoints,
                 covariance_estimator=covariance)
    return model.eval()
