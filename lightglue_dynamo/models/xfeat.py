"""
XFeat: Accelerated Features for Lightweight Image Matching (CVPR 2024).
Native port of the minimal forward pass — no ref/accelerated_features dependency.
Only detectAndCompute() is retained; match/dense/kornia paths are dropped.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, stride=stride,
                      dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class _XFeatModel(nn.Module):
    """XFeat backbone: returns (feats B×64×H/8×W/8, keypoints B×65×H/8×W/8, heatmap B×1×H/8×W/8)."""

    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(1, 24, 1, stride=1, padding=0),
        )

        self.block1 = nn.Sequential(
            _BasicLayer(1,  4, stride=1),
            _BasicLayer(4,  8, stride=2),
            _BasicLayer(8,  8, stride=1),
            _BasicLayer(8, 24, stride=2),
        )

        self.block2 = nn.Sequential(
            _BasicLayer(24, 24, stride=1),
            _BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            _BasicLayer(24, 64, stride=2),
            _BasicLayer(64, 64, stride=1),
            _BasicLayer(64, 64, 1, padding=0),
        )

        self.block4 = nn.Sequential(
            _BasicLayer(64, 64, stride=2),
            _BasicLayer(64, 64, stride=1),
            _BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            _BasicLayer( 64, 128, stride=2),
            _BasicLayer(128, 128, stride=1),
            _BasicLayer(128, 128, stride=1),
            _BasicLayer(128,  64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            _BasicLayer(64, 64, stride=1),
            _BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0),
        )

        self.heatmap_head = nn.Sequential(
            _BasicLayer(64, 64, 1, padding=0),
            _BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        self.keypoint_head = nn.Sequential(
            _BasicLayer(64, 64, 1, padding=0),
            _BasicLayer(64, 64, 1, padding=0),
            _BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        x = self.norm(x)

        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)

        heatmap = self.heatmap_head(feats)
        # pixel_unshuffle is ONNX-safe and equivalent to the original _unfold2d(x, ws=8)
        keypoints = self.keypoint_head(F.pixel_unshuffle(x, 8))

        return feats, keypoints, heatmap


class _InterpolateSparse2d(nn.Module):
    """Efficiently interpolate tensor at given sparse 2D positions."""

    def __init__(self, mode='bicubic', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        return 2. * (x / (torch.tensor([W - 1, H - 1], device=x.device, dtype=x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode=self.mode, align_corners=False)
        return x.permute(0, 2, 3, 1).squeeze(-2)


class XFeat(nn.Module):
    """
    XFeat sparse feature extractor (CVPR 2024).

    Self-contained; no dependency on ref/accelerated_features.
    API-compatible with the ref XFeat class for use with XFeatPaddedOnnxWrapper.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        top_k: int = 4096,
        detection_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = _XFeatModel().to(self.dev).eval()
        self.top_k = top_k
        self.detection_threshold = detection_threshold

        if weights is not None:
            if isinstance(weights, (str, Path)):
                self.net.load_state_dict(
                    torch.load(str(weights), map_location=self.dev)
                )
            else:
                self.net.load_state_dict(weights)

        self.interpolator = _InterpolateSparse2d('bicubic')

    def detectAndCompute(self, x, top_k=None, detection_threshold=None):
        """
        Compute sparse keypoints & descriptors. Supports batched mode.

        Input:
            x -> torch.Tensor(B, C, H, W): grayscale or rgb image
            top_k -> int: keep best k features
        Return:
            List[Dict]:
                'keypoints'    -> torch.Tensor(N, 2): keypoints (x, y)
                'scores'       -> torch.Tensor(N,): keypoint scores
                'descriptors'  -> torch.Tensor(N, 64): local features
        """
        if top_k is None:
            top_k = self.top_k
        if detection_threshold is None:
            detection_threshold = self.detection_threshold
        x, rh1, rw1 = self.preprocess_tensor(x)

        B, _, _H1, _W1 = x.shape

        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)

        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5)

        _nearest = _InterpolateSparse2d('nearest')
        _bilinear = _InterpolateSparse2d('bilinear')
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :top_k]

        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)
        feats = F.normalize(feats, dim=-1)

        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)

        valid = scores > 0
        return [
            {
                'keypoints': mkpts[b][valid[b]],
                'scores': scores[b][valid[b]],
                'descriptors': feats[b][valid[b]],
            }
            for b in range(B)
        ]

    def preprocess_tensor(self, x):
        """Guarantee that image is divisible by 32 to avoid aliasing artifacts."""
        if len(x.shape) != 4:
            raise RuntimeError('Input tensor needs to be in (B,C,H,W) format')
        x = x.to(self.dev).float()
        H, W = x.shape[-2:]
        _H, _W = (H // 32) * 32, (W // 32) * 32
        rh, rw = H / _H, W / _W
        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap

    def NMS(self, x, threshold=0.05, kernel_size=5):
        B, _, H, W = x.shape
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(p) for p in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos

    def forward_export(
        self, images: torch.Tensor, top_k: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched fixed-K extraction — fully ONNX-traceable.

        Replaces the per-image NMS+nonzero loop with a batched topk over the score map.
        Input must be pre-padded to multiples of 32 (guaranteed by the registry preprocessor).

        Returns: keypoints (B,K,2), scores (B,K), descriptors (B,K,64), num_valid (B,)
        """
        if top_k is None:
            top_k = self.top_k

        x = images.float()
        B, _, H, W = x.shape

        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)

        K1h = self.get_kpts_heatmap(K1)      # (B, 1, H, W)
        fH, fW = K1h.shape[-2:]

        # ONNX-safe NMS: max-pool peak mask + threshold, no nonzero()
        local_max = F.max_pool2d(K1h, kernel_size=5, stride=1, padding=2)
        is_peak = ((K1h == local_max) & (K1h > self.detection_threshold)).float()

        # Reliability-weighted score map
        H1_up = F.interpolate(H1, size=(fH, fW), mode='bilinear', align_corners=False)
        scores_map = (K1h * is_peak * H1_up).squeeze(1)   # (B, fH, fW)

        # Fixed-K topk selection
        scores_flat = scores_map.reshape(B, fH * fW)
        top_scores, top_inds = torch.topk(scores_flat, k=top_k, dim=1)  # (B, K)

        # Convert flat indices → (x, y) pixel coords in [0,W-1] x [0,H-1]
        y_inds = (top_inds // fW).float()
        x_inds = (top_inds % fW).float()
        mkpts = torch.stack([x_inds, y_inds], dim=-1)   # (B, K, 2)

        # Interpolate descriptors from M1 (at H//8 × W//8) using full-res coords
        feats = self.interpolator(M1, mkpts, H=fH, W=fW)
        feats = F.normalize(feats, dim=-1)

        num_valid = (top_scores > 0).sum(dim=1).to(torch.int32)
        return mkpts, top_scores, feats, num_valid


__all__ = ["XFeat"]
