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

    # def forward(self, data):
    #     image = data["image"]
    #     if image.shape[1] == 3:  # RGB to gray
    #         scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    #         image = (image * scale).sum(1, keepdim=True)

    #     features = self.backbone(image)
    #     descriptors_dense = torch.nn.functional.normalize(
    #         self.descriptor(features), p=2, dim=1
    #     )

    #     # Decode the detection scores
    #     scores = self.detector(features)
    #     scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
    #     b, _, h, w = scores.shape
    #     # scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
    #     # scores = scores.permute(0, 1, 3, 2, 4).reshape(
    #     #     b, h * self.stride, w * self.stride
    #     # )
    #     # scores = batched_nms(scores, self.conf.nms_radius)
    #     scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
    #     scores = scores.permute(0, 1, 3, 2, 4).reshape(b, 1, h * self.stride, w * self.stride)
    #     scores = batched_nms(scores, self.conf.nms_radius)

    #     # Discard keypoints near the image borders
    #     # if self.conf.remove_borders:
    #     #     pad = self.conf.remove_borders
    #     #     scores[:, :pad] = -1
    #     #     scores[:, :, :pad] = -1
    #     #     scores[:, -pad:] = -1
    #     #     scores[:, :, -pad:] = -1
    #     if self.conf.remove_borders:         
    #         pad = self.conf.remove_borders     # scores shape is (b, 1, H, W) after the reshape above
    #         scores[:, :, :pad, :] = -1         # top rows
    #         scores[:, :, :, :pad] = -1         # left cols            
    #         scores[:, :, -pad:, :] = -1        # bottom rows
    #         scores[:, :, :, -pad:] = -1        # right cols

    #     # Extract keypoints
    #     if b > 1:
    #         idxs = torch.where(scores > self.conf.detection_threshold)
    #         mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
    #     else:  # Faster shortcut
    #         scores = scores.squeeze(0).squeeze(0)
    #         idxs = torch.where(scores > self.conf.detection_threshold)

    #     # Convert (i, j) to (x, y)
    #     keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
    #     scores_all = scores[idxs]

    #     keypoints = []
    #     scores = []
    #     descriptors = []
    #     for i in range(b):
    #         if b > 1:
    #             k = keypoints_all[mask[i]]
    #             s = scores_all[mask[i]]
    #         else:
    #             k = keypoints_all
    #             s = scores_all
    #         if self.conf.max_num_keypoints is not None:
    #             k, s = select_top_k_keypoints(k, s, self.conf.max_num_keypoints)
    #         d = sample_descriptors(k[None], descriptors_dense[i, None], self.stride)
    #         keypoints.append(k)
    #         scores.append(s)
    #         descriptors.append(d.squeeze(0).transpose(0, 1))

    #     return {
    #         "keypoints": keypoints,
    #         "keypoint_scores": scores,
    #         "descriptors": descriptors,
    #     }

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
            scores[:, :, :pad, :] = -1
            scores[:, :, :, :pad] = -1
            scores[:, :, -pad:, :] = -1
            scores[:, :, :, -pad:] = -1

        max_kpts = self.conf.max_num_keypoints
        if max_kpts is None:
            raise ValueError("max_num_keypoints must be set in the config for batched export.")

        device = scores.device
        # Shape: (batch_size, max_kpts, 2) for (x, y) coordinates
        keypoints_padded = torch.zeros((b, max_kpts, 2), device=device, dtype=torch.float)
        # Shape: (batch_size, max_kpts)
        scores_padded = torch.zeros((b, max_kpts), device=device, dtype=torch.float)
        # Shape: (batch_size, descriptor_dim, max_kpts)
        desc_dim = descriptors_dense.shape[1]
        descriptors_padded = torch.zeros((b, desc_dim, max_kpts), device=device, dtype=torch.float)
        # Shape: (batch_size,) to store the actual number of keypoints found
        num_keypoints = torch.zeros(b, device=device, dtype=torch.long)

        idxs = torch.where(scores > self.conf.detection_threshold)
        
        if idxs[0].shape[0] == 0:
            return {
                "keypoints": keypoints_padded,
                "keypoint_scores": scores_padded,
                "descriptors": descriptors_padded.transpose(-1, -2),
                "num_keypoints": num_keypoints,
            }

        # Create a mask to group keypoints by their image index in the batch
        batch_idx_tensor = idxs[0]
        mask = batch_idx_tensor == torch.arange(b, device=device)[:, None]

        # Get all keypoints and scores from the batch
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        for i in range(b):
            # Select keypoints and scores for the i-th image
            k_i = keypoints_all[mask[i]]
            s_i = scores_all[mask[i]]

            if k_i.shape[0] == 0:
                continue
            
            # Apply Top-K selection
            k_i, s_i = select_top_k_keypoints(k_i, s_i, max_kpts)
            
            # Store the actual number of keypoints found
            n_kpts = k_i.shape[0]
            num_keypoints[i] = n_kpts
            
            # Fill the pre-allocated tensors
            keypoints_padded[i, :n_kpts] = k_i
            scores_padded[i, :n_kpts] = s_i
            
            # Sample descriptors for the selected keypoints
            d_i = sample_descriptors(k_i[None], descriptors_dense[i, None], self.stride)
            descriptors_padded[i, :, :n_kpts] = d_i.squeeze(0)

        return {
            "keypoints": keypoints_padded,
            "keypoint_scores": scores_padded,
            "descriptors": descriptors_padded.transpose(-1, -2), 
            "num_keypoints": num_keypoints,
        }