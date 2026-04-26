"""ONNX validation helpers: checker and operator inventory."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx


def onnx_op_type_counts(model_path: str | Path) -> Counter[str]:
    import onnx

    m = onnx.load_model(str(model_path))
    return Counter(n.op_type for n in m.graph.node)


def validate_onnx(model_path: str | Path, *, run_shape_inference: bool = True) -> None:
    import onnx
    from onnx import checker

    path = Path(model_path)
    model = onnx.load_model(str(path))
    checker.check_model(model)
    if run_shape_inference:
        try:
            inferred = onnx.shape_inference.infer_shapes(model)
            onnx.save_model(inferred, str(path))
        except Exception:
            pass
