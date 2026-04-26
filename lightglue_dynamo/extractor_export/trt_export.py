"""onnxsim simplification and TensorRT engine build for extractor-only ONNX models."""

from __future__ import annotations

from pathlib import Path


def simplify_onnx(onnx_path: str | Path, output_path: str | Path) -> Path:
    """Simplify an ONNX model with onnxsim.

    Raises AssertionError if the simplification check fails.
    """
    try:
        from onnxsim import simplify
    except ImportError as exc:
        raise ImportError("onnxsim is required: pip install onnxsim") from exc

    import onnx

    model = onnx.load(str(onnx_path))
    model_sim, check = simplify(model)
    if not check:
        raise AssertionError(f"onnxsim validation failed for {onnx_path}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_sim, str(out))
    return out


def build_extractor_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path | None = None,
    *,
    fp16: bool = True,
) -> Path:
    """Build a TensorRT engine from an extractor ONNX model using Polygraphy.

    Parameters
    ----------
    onnx_path:
        Path to the (preferably onnxsim-simplified) ONNX model.
    engine_path:
        Output path for the ``.engine`` file. Defaults to ``onnx_path`` with
        ``.engine`` suffix replacing the original suffix.
    fp16:
        Enable FP16 precision in TensorRT (recommended for Jetson / NVIDIA GPUs).
    """
    try:
        from polygraphy.backend.trt import (
            CreateConfig,
            EngineFromNetwork,
            NetworkFromOnnxPath,
            SaveEngine,
            TrtRunner,
        )
    except ImportError as exc:
        raise ImportError(
            "polygraphy and tensorrt are required: pip install polygraphy tensorrt"
        ) from exc

    onnx_path = Path(onnx_path)
    if engine_path is None:
        engine_path = onnx_path.with_suffix(".engine")
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    build_engine = SaveEngine(
        EngineFromNetwork(
            NetworkFromOnnxPath(str(onnx_path)),
            config=CreateConfig(fp16=fp16),
        ),
        str(engine_path),
    )
    with TrtRunner(build_engine):
        pass

    return engine_path
