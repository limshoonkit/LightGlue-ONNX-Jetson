"""PyTorch implementation of the SuperPoint model,
   derived from the TensorFlow re-implementation (2018).
   Authors: Rémi Pautrat, Paul-Edouard Sarlin
"""
import torch.nn as nn
import torch
from collections import OrderedDict
from types import SimpleNamespace


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def batched_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius, dilation=1
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding
        )
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(
            OrderedDict(
                [
                    ("conv", conv),
                    ("activation", activation),
                    ("bn", bn),
                ]
            )
        )


class SuperPointOpen(nn.Module):
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )
    
    def forward(self, data):
        image = data["image"]
        if image.shape[1] == 3:  # RGB to gray
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        features = self.backbone(image)
        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )

        scores = self.detector(features)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, 1, h * self.stride, w * self.stride
        )
        scores = batched_nms(scores, self.conf.nms_radius)

        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :, :pad, :] = 0
            scores[:, :, :, :pad] = 0
            scores[:, :, -pad:, :] = 0
            scores[:, :, :, -pad:] = 0

        max_kpts = self.conf.max_num_keypoints
        if max_kpts is None:
            raise ValueError("max_num_keypoints must be set for DLA-friendly export.")

        # 1. Flatten scores and get the top K scores and their flat indices
        scores_flat = scores.reshape(b, -1)
        topk_scores, topk_indices = torch.topk(scores_flat, k=max_kpts, dim=1)

        # 2. Convert flat indices back to 2D coordinates (x, y)
        img_h, img_w = scores.shape[2], scores.shape[3]
        topk_y = topk_indices // img_w
        topk_x = topk_indices % img_w
        
        # Shape: (b, max_kpts, 2)
        keypoints = torch.stack([topk_x, topk_y], dim=-1).float()

        # 3. Apply the detection threshold
        threshold_mask = (topk_scores > self.conf.detection_threshold).float()
        
        final_scores = topk_scores * threshold_mask
        # Unsqueeze to broadcast the mask over the (x,y) coordinates
        final_keypoints = keypoints * threshold_mask.unsqueeze(-1)
        
        # 4. Count the number of valid keypoints per image
        num_keypoints = torch.sum(threshold_mask, dim=1, dtype=torch.long)

        # 5. Sample descriptors in a fully batched operation
        # This avoids the Python for-loop and dynamic indexing.
        descriptors = sample_descriptors(
            final_keypoints, descriptors_dense, self.stride
        ) # output shape is (b, desc_dim, max_kpts)
        
        # Mask out descriptors for filtered keypoints
        # Permute mask to (b, 1, max_kpts) to broadcast over descriptor dim
        final_descriptors = descriptors * threshold_mask.unsqueeze(1)

        return {
            "keypoints": final_keypoints,
            "keypoint_scores": final_scores,
            "descriptors": final_descriptors.transpose(-1, -2),
            "num_keypoints": num_keypoints,
        }
    
    def forward_for_export(self, image):
        result = self.forward({"image": image})
        return (
            result["keypoints"],
            result["keypoint_scores"], 
            result["descriptors"],
            result["num_keypoints"]
        )
