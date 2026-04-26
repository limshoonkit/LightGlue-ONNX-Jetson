"""Register torchvision DeformConv for ONNX export (ALIKED)."""

from __future__ import annotations


def register_deform_conv2d_onnx(opset: int = 19) -> None:
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import _get_const

    def deform_conv2d_symbolic(
        g,
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    ):
        stride = [_get_const(stride_h, "i", "stride_h"), _get_const(stride_w, "i", "stride_w")]
        padding = [_get_const(pad_h, "i", "pad_h"), _get_const(pad_w, "i", "pad_w")]
        dilations = [_get_const(dil_h, "i", "dil_h"), _get_const(dil_w, "i", "dil_w")]
        groups = _get_const(n_weight_grps, "i", "n_weight_grps")
        offset_groups = _get_const(n_offset_grps, "i", "n_offset_grps")
        pads = [padding[0], padding[1], padding[0], padding[1]]
        if mask.node().mustBeNone():
            mask = g.op("Constant")
        if bias.node().mustBeNone():
            bias = g.op("Constant")
        return g.op(
            "DeformConv",
            input,
            weight,
            offset,
            mask,
            bias,
            strides_i=stride,
            pads_i=pads,
            dilations_i=dilations,
            group_i=groups,
            offset_group_i=offset_groups,
        )

    register_custom_op_symbolic("torchvision::deform_conv2d", deform_conv2d_symbolic, opset)
