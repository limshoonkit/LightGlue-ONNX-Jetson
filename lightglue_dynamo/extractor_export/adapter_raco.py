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
    """

    def __init__(self, raco_module: nn.Module) -> None:
        super().__init__()
        self.raco = raco_module

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.raco.forward({"image": images})
        return out["keypoints"], out["keypoint_scores"]


def load_raco(weights_path: str | Path, *, max_keypoints: int = 2048) -> nn.Module:
    from lightglue_dynamo.models.raco import RaCo

    model = RaCo(weights=str(weights_path), max_num_keypoints=max_keypoints)
    return model.eval()
