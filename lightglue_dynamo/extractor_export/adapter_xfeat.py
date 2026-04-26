"""XFeat ONNX adapter: per-image detectAndCompute with fixed-K padding (batched export)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class XFeatPaddedOnnxWrapper(nn.Module):
    """
    Thin ONNX wrapper around ``XFeat.forward_export``.

    Delegates to the batched, topk-based forward that is fully ONNX-traceable
    (no nonzero / dynamic shapes). Input: ``(B, C, H, W)`` float [0,1].
    """

    def __init__(self, xfeat_core: nn.Module, max_k: int) -> None:
        super().__init__()
        self.xfeat = xfeat_core
        self.max_k = max_k

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.xfeat.forward_export(images, top_k=self.max_k)


def load_xfeat(weights_path: str | Path, *, device: torch.device | str = "cpu") -> nn.Module:
    from lightglue_dynamo.models.xfeat import XFeat

    dev = torch.device(device) if isinstance(device, str) else device
    model = XFeat(weights=str(weights_path), top_k=4096)
    model.dev = dev
    model.net = model.net.to(dev)
    return model.eval()
