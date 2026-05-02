import torch
from torch import nn, Tensor

# coordinates system
#  ------------------------------>  [ x: range=-1.0~1.0; w: range=0~W ]
#  | -----------------------------
#  | |                           |
#  | |                           |
#  | |                           |
#  | |         image             |
#  | |                           |
#  | |                           |
#  | |                           |
#  | |---------------------------|
#  v
# [ y: range=-1.0~1.0; h: range=0~H ]


def simple_nms(scores: Tensor, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""

    zeros = torch.zeros_like(scores)
    max_mask = scores == torch.nn.functional.max_pool2d(
        scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
    )

    for _ in range(2):
        supp_mask = (
            torch.nn.functional.max_pool2d(
                max_mask.float(),
                kernel_size=nms_radius * 2 + 1,
                stride=1,
                padding=nms_radius,
            )
            > 0
        )
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == torch.nn.functional.max_pool2d(
            supp_scores,
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        )
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class DKD(nn.Module):
    def __init__(
        self,
        radius: int = 2,
        top_k: int = 0,
        scores_th: float = 0.2,
        n_limit: int = 20000,
        script: bool = False,
    ):
        """
        Args:
            radius: soft detection radius, kernel size is (2 * radius + 1)
            top_k: top_k > 0: return top k keypoints
            scores_th: top_k <= 0 threshold mode:  scores_th > 0: return keypoints with scores>scores_th
                                                   else: return keypoints with scores > scores.mean()
            n_limit: max number of keypoint in threshold mode
        """
        super().__init__()
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.kernel_size = 2 * self.radius + 1
        self.temperature = 0.1  # tuned temperature
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size, padding=self.radius
        )
        # self.get_patches_func = get_patches_script if script else get_patches.apply

        # local xy grid (buffer follows module device; avoids hard-coded CUDA)
        lin = torch.linspace(-self.radius, self.radius, self.kernel_size)
        try:
            g0, g1 = torch.meshgrid(lin, lin, indexing="ij")
        except TypeError:
            g0, g1 = torch.meshgrid(lin, lin)
        hw = torch.stack((g0, g1)).view(2, -1).t()[:, [1, 0]]
        self.register_buffer("hw_grid", hw, persistent=False)

    def detect_keypoints(self, scores_map: Tensor, sub_pixel: bool = True):
        b, c, h, w = scores_map.shape
        scores_nograd = scores_map.detach()
        nms_scores = simple_nms(scores_nograd, 2)

        # remove border
        nms_scores[:, :, : self.radius, :] = 0
        nms_scores[:, :, :, : self.radius] = 0
        nms_scores[:, :, h - self.radius :, :] = 0
        nms_scores[:, :, :, w - self.radius :] = 0

        # Fast path used by the ONNX export (configure_aliked_for_export forces top_k > 0):
        # everything stays as (B, K, ...) tensors, so TRT can batch the postproc and grid_sample
        # collapses to a single op over (B, 1, H, W) sampled at (B, 1, K, 2).
        if self.top_k > 0:
            return self._detect_keypoints_topk_batched(scores_map, scores_nograd, nms_scores, sub_pixel)

        if self.scores_th > 0:
            masks = nms_scores > self.scores_th
            if masks.sum() == 0:
                th = scores_nograd.reshape(b, -1).mean(dim=1)
                masks = nms_scores > th.reshape(b, 1, 1, 1)
        else:
            th = scores_nograd.reshape(b, -1).mean(dim=1)
            masks = nms_scores > th.reshape(b, 1, 1, 1)
        masks = masks.reshape(b, -1)

        indices_keypoints = []  # list, B x (any size)
        scores_view = scores_nograd.reshape(b, -1)
        for mask, scores in zip(masks, scores_view):
            indices = mask.nonzero()[:, 0]
            if len(indices) > self.n_limit:
                kpts_sc = scores[indices]
                sort_idx = kpts_sc.sort(descending=True)[1]
                sel_idx = sort_idx[: self.n_limit]
                indices = indices[sel_idx]
            indices_keypoints.append(indices)

        wh = torch.tensor([w - 1, h - 1], device=scores_nograd.device)

        keypoints = []
        scoredispersitys = []
        kptscores = []
        if sub_pixel:
            # detect soft keypoints with grad backpropagation
            patches = self.unfold(scores_map)  # B x (kernel**2) x (H*W)
            self.hw_grid = self.hw_grid.to(scores_map)  # to device
            for b_idx in range(b):
                patch = patches[b_idx].t()  # (H*W) x (kernel**2)
                indices_kpt = indices_keypoints[
                    b_idx
                ]  # one dimension vector, say its size is M
                patch_scores = patch[indices_kpt]  # M x (kernel**2)
                keypoints_xy_nms = torch.stack(
                    [
                        indices_kpt % w,
                        torch.div(indices_kpt, w, rounding_mode="trunc"),
                    ],
                    dim=1,
                )  # Mx2

                # max is detached to prevent undesired backprop loops in the graph
                max_v = patch_scores.max(dim=1).values.detach()[:, None]
                x_exp = (
                    (patch_scores - max_v) / self.temperature
                ).exp()  # M * (kernel**2), in [0, 1]

                # \frac{ \sum{(i,j) \times \exp(x/T)} }{ \sum{\exp(x/T)} }
                xy_residual = (
                    x_exp @ self.hw_grid / x_exp.sum(dim=1)[:, None]
                )  # Soft-argmax, Mx2

                hw_grid_dist2 = (
                    torch.norm(
                        (self.hw_grid[None, :, :] - xy_residual[:, None, :])
                        / self.radius,
                        dim=-1,
                    )
                    ** 2
                )
                scoredispersity = (x_exp * hw_grid_dist2).sum(
                    dim=1
                ) / x_exp.sum(dim=1)

                # compute result keypoints
                keypoints_xy = keypoints_xy_nms + xy_residual
                keypoints_xy = (
                    keypoints_xy / wh * 2 - 1
                )  # (w,h) -> (-1~1,-1~1)

                kptscore = torch.nn.functional.grid_sample(
                    scores_map[b_idx].unsqueeze(0),
                    keypoints_xy.view(1, 1, -1, 2),
                    mode="bilinear",
                    align_corners=True,
                )[
                    0, 0, 0, :
                ]  # CxN

                keypoints.append(keypoints_xy)
                scoredispersitys.append(scoredispersity)
                kptscores.append(kptscore)
        else:
            for b_idx in range(b):
                indices_kpt = indices_keypoints[
                    b_idx
                ]  # one dimension vector, say its size is M
                # To avoid warning: UserWarning: __floordiv__ is deprecated
                keypoints_xy_nms = torch.stack(
                    [
                        indices_kpt % w,
                        torch.div(indices_kpt, w, rounding_mode="trunc"),
                    ],
                    dim=1,
                )  # Mx2
                keypoints_xy = (
                    keypoints_xy_nms / wh * 2 - 1
                )  # (w,h) -> (-1~1,-1~1)
                kptscore = torch.nn.functional.grid_sample(
                    scores_map[b_idx].unsqueeze(0),
                    keypoints_xy.view(1, 1, -1, 2),
                    mode="bilinear",
                    align_corners=True,
                )[
                    0, 0, 0, :
                ]  # CxN
                keypoints.append(keypoints_xy)
                scoredispersitys.append(
                    kptscore
                )  # for jit.script compatability
                kptscores.append(kptscore)

        return keypoints, scoredispersitys, kptscores

    def _detect_keypoints_topk_batched(
        self,
        scores_map: Tensor,
        scores_nograd: Tensor,
        nms_scores: Tensor,
        sub_pixel: bool,
    ):
        """Vectorised replacement for the per-batch loop when top_k > 0.

        Returns the same (keypoints, scoredispersitys, kptscores) list-of-tensors API as
        detect_keypoints;
        """
        b, _, h, w = scores_map.shape
        K = self.top_k
        kk = self.kernel_size * self.kernel_size

        # (B, K) flat indices into the score map. Stays batched throughout.
        indices_b = torch.topk(nms_scores.view(b, -1), K).indices  # (B, K)

        ix = (indices_b % w).to(scores_map.dtype)
        iy = torch.div(indices_b, w, rounding_mode="trunc").to(scores_map.dtype)
        keypoints_xy_nms_b = torch.stack([ix, iy], dim=-1)  # (B, K, 2)

        wh = torch.tensor([w - 1, h - 1], device=scores_nograd.device, dtype=scores_map.dtype)

        if sub_pixel:
            self.hw_grid = self.hw_grid.to(scores_map)

            # patches: (B, k^2, H*W). Gather K columns per batch in one op.
            patches = self.unfold(scores_map)
            gather_idx = indices_b.unsqueeze(1).expand(-1, kk, -1)  # (B, k^2, K)
            patch_scores = patches.gather(2, gather_idx).transpose(1, 2)  # (B, K, k^2)

            # max detached to avoid backprop loops (matches original code).
            max_v = patch_scores.max(dim=-1, keepdim=True).values.detach()  # (B, K, 1)
            x_exp = ((patch_scores - max_v) / self.temperature).exp()       # (B, K, k^2)
            x_exp_sum = x_exp.sum(dim=-1, keepdim=True)                     # (B, K, 1)

            # Soft-argmax: (B, K, k^2) @ (k^2, 2) / (B, K, 1) -> (B, K, 2)
            xy_residual_b = (x_exp @ self.hw_grid) / x_exp_sum

            # Score dispersity, batched over (B, K).
            hw_grid_dist2 = (
                ((self.hw_grid[None, None, :, :] - xy_residual_b[:, :, None, :]) / self.radius)
                .norm(dim=-1) ** 2
            )  # (B, K, k^2)
            scoredispersity_b = (x_exp * hw_grid_dist2).sum(dim=-1) / x_exp_sum.squeeze(-1)

            keypoints_xy_b = (keypoints_xy_nms_b + xy_residual_b) / wh * 2 - 1  # (B, K, 2)
        else:
            keypoints_xy_b = keypoints_xy_nms_b / wh * 2 - 1
            scoredispersity_b = None  # filled below for jit.script compat

        # Single batched grid_sample: (B, 1, H, W) sampled at (B, 1, K, 2) -> (B, 1, 1, K)
        kptscore_b = torch.nn.functional.grid_sample(
            scores_map,
            keypoints_xy_b.unsqueeze(1),
            mode="bilinear",
            align_corners=True,
        ).view(b, K)

        if scoredispersity_b is None:
            scoredispersity_b = kptscore_b  # match original sub_pixel=False placeholder

        keypoints = [keypoints_xy_b[i] for i in range(b)]
        kptscores = [kptscore_b[i] for i in range(b)]
        scoredispersitys = [scoredispersity_b[i] for i in range(b)]
        return keypoints, scoredispersitys, kptscores

    def forward(self, scores_map: Tensor, sub_pixel: bool = True):
        """
        :param scores_map: Bx1xHxW
        :param descriptor_map: BxCxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1.0 ~ 1.0
        """

        keypoints, scoredispersitys, kptscores = self.detect_keypoints(
            scores_map, sub_pixel
        )

        return keypoints, kptscores, scoredispersitys