"""Torch modules that reshape extractor outputs for fixed-K ONNX export."""

from __future__ import annotations

import torch
import torch.nn as nn

from lightglue_dynamo.models.superpoint_pytorch import SuperPointOpen


class SuperPointOpenOnnxWrapper(nn.Module):
    """Maps dict output to a flat tuple for ONNX."""

    def __init__(self, model: SuperPointOpen) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self.model.eval()
        preds = self.model({"image": images})
        return (
            preds["keypoints"],
            preds["keypoint_scores"],
            preds["descriptors"],
            preds["num_keypoints"],
        )


class AlikedOnnxWrapper(nn.Module):
    """Pads variable-length ALIKED outputs to fixed (B, max_k, ...)."""

    def __init__(self, model: nn.Module, max_kps: int) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.max_kps = max_kps

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        preds = self.model.forward(image)
        kpts_list: list[torch.Tensor] = []
        scores_list: list[torch.Tensor] = []
        desc_list: list[torch.Tensor] = []
        batch_size = image.shape[0]
        for i in range(batch_size):
            kpts = preds["keypoints"][i]
            scores = preds["scores"][i]
            descriptors = preds["descriptors"][i]
            num_detected = kpts.shape[0]
            if num_detected > self.max_kps:
                top_indices = torch.topk(scores, self.max_kps, sorted=False)[1]
                kpts = kpts[top_indices]
                scores = scores[top_indices]
                descriptors = descriptors[top_indices]
                num_detected = self.max_kps
            kpts_padded = torch.zeros(self.max_kps, 2, device=kpts.device, dtype=kpts.dtype)
            scores_padded = torch.zeros(self.max_kps, device=scores.device, dtype=scores.dtype)
            desc_padded = torch.zeros(
                self.max_kps, descriptors.shape[1], device=descriptors.device, dtype=descriptors.dtype
            )
            if num_detected > 0:
                kpts_padded[:num_detected] = kpts
                scores_padded[:num_detected] = scores
                desc_padded[:num_detected] = descriptors
            kpts_list.append(kpts_padded)
            scores_list.append(scores_padded)
            desc_list.append(desc_padded)
        final_kpts = torch.stack(kpts_list, dim=0)
        final_scores = torch.stack(scores_list, dim=0)
        final_descriptors = torch.stack(desc_list, dim=0)
        num_valid_keypoints = torch.tensor(
            [min(int(preds["keypoints"][i].shape[0]), self.max_kps) for i in range(batch_size)],
            device=image.device,
            dtype=torch.int32,
        )
        return final_kpts, final_scores, final_descriptors, num_valid_keypoints


def configure_aliked_for_export(extractor: nn.Module, max_kps: int) -> nn.Module:
    if hasattr(extractor, "dkd"):
        extractor.dkd.n_limit = max_kps
        if extractor.dkd.top_k <= 0:
            extractor.dkd.top_k = max_kps
        else:
            extractor.dkd.top_k = min(int(extractor.dkd.top_k), max_kps)
        extractor.dkd.scores_th = max(float(extractor.dkd.scores_th), 0.25)
    return extractor


class SuperPointCvgOnnxWrapper(nn.Module):
    """Wraps cvg SuperPoint tuple (kpts, scores, desc) with consistent output names."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        k, s, d = self.model(images)
        return k, s, d


class DiskOnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        k, s, d = self.model(images)
        return k, s, d


class AlikedDescribeWrapper(nn.Module):
    """ALIKED-n16 describe-at-external-keypoints for fixed-K ONNX export.

    Inputs
    ------
    image  : (B, 3, H, W)  float32, RGB in [0, 1]
    kpts_n : (B, K, 2)     keypoints anisotropically normalised to [-1, 1]
                            i.e.  kpts_n = 2 * kpts_px / [W-1, H-1] - 1

    Output
    ------
    descriptors : (B, K, 128)  L2-normalised ALIKED/SDDH descriptors
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, kpts_n: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        feature_map, _ = self.model.extract_dense_map(image)
        descs_list, _ = self.model.desc_head(feature_map, [kpts_n[i] for i in range(b)])
        return torch.stack(descs_list, dim=0)
