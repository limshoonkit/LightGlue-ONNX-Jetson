from pathlib import Path
from typing import Annotated, Optional

import cv2
import typer

from lightglue_dynamo.cli_utils import check_multiple_of
from lightglue_dynamo.config import Extractor, InferenceDevice

app = typer.Typer()

import numpy as np
from pathlib import Path
from typing import Optional, Annotated, Iterable, Dict, Any

from lightglue_dynamo.config import Extractor
from lightglue_dynamo.preprocessors import DISKPreprocessor, SuperPointPreprocessor

def calib_data_loader(
    calib_dir: Path, height: int, width: int, extractor_type: Extractor
) -> Iterable[Dict[str, Any]]:
    """
    Generator that loads and preprocesses calibration image pairs.
    Yields data in the format required by Polygraphy's Calibrator.
    """
    print(f"Loading calibration data from: {calib_dir}")
    left_images = sorted(calib_dir.glob("*_left.png"))
    if not left_images:
        raise ValueError(f"No '*_left.png' files found in {calib_dir}")

    for left_path in left_images:
        right_path = left_path.parent / left_path.name.replace("_left.png", "_right.png")
        if not right_path.exists():
            print(f"Warning: Corresponding right image not found for {left_path.name}, skipping pair.")
            continue

        raw_images = [
            cv2.resize(cv2.imread(str(p)), (width, height)) for p in [left_path, right_path]
        ]
        images = np.stack(raw_images)

        # Apply the same preprocessing as the main inference path
        match extractor_type:
            case Extractor.superpoint:
                images = SuperPointPreprocessor.preprocess(images)
            case Extractor.disk:
                images = DISKPreprocessor.preprocess(images)
        images = images.astype(np.float32)

        # Polygraphy calibrator expects a dictionary mapping input names to numpy arrays
        yield {"images": images}

@app.callback()
def callback():
    """LightGlue Dynamo CLI"""


@app.command()
def export(
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output: Annotated[
        Optional[Path],  # typer does not support Path | None # noqa: UP007
        typer.Option("-o", "--output", dir_okay=False, writable=True, help="Path to save exported model."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "-b", "--batch-size", min=0, help="Batch size of exported ONNX model. Set to 0 to mark as dynamic."
        ),
    ] = 0,
    height: Annotated[
        int, typer.Option("-h", "--height", min=0, help="Height of input image. Set to 0 to mark as dynamic.")
    ] = 0,
    width: Annotated[
        int, typer.Option("-w", "--width", min=0, help="Width of input image. Set to 0 to mark as dynamic.")
    ] = 0,
    num_keypoints: Annotated[
        int, typer.Option(min=128, help="Number of keypoints outputted by feature extractor.")
    ] = 1024,
    fuse_multi_head_attention: Annotated[
        bool,
        typer.Option(
            "--fuse-multi-head-attention",
            help="Fuse multi-head attention subgraph into one optimized operation. (ONNX Runtime-only).",
        ),
    ] = False,
    opset: Annotated[int, typer.Option(min=16, max=20, help="ONNX opset version of exported model.")] = 19,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether to also convert to FP16.")] = False,
):
    """Export LightGlue to ONNX."""
    import onnx
    import torch
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    from onnxruntime.transformers.float16 import convert_float_to_float16

    from lightglue_dynamo.models import DISK, LightGlue, Pipeline, SuperPoint
    from lightglue_dynamo.ops import use_fused_multi_head_attention

    match extractor_type:
        case Extractor.superpoint:
            extractor = SuperPoint(num_keypoints=num_keypoints)
        case Extractor.disk:
            extractor = DISK(num_keypoints=num_keypoints)
    matcher = LightGlue(**extractor_type.lightglue_config)
    pipeline = Pipeline(extractor, matcher).eval()

    if output is None:
        output = Path(f"weights/{extractor_type}_lightglue_pipeline.onnx")

    check_multiple_of(batch_size, 2)
    check_multiple_of(height, extractor_type.input_dim_divisor)
    check_multiple_of(width, extractor_type.input_dim_divisor)

    if height > 0 and width > 0 and num_keypoints > height * width:
        raise typer.BadParameter("num_keypoints cannot be greater than height * width.")

    if fuse_multi_head_attention:
        typer.echo(
            "Warning: Multi-head attention nodes will be fused. Exported model will only work with ONNX Runtime CPU & CUDA execution providers."
        )
        if torch.__version__ < "2.4":
            raise typer.Abort("Fused multi-head attention requires PyTorch 2.4 or later.")
        use_fused_multi_head_attention()

    dynamic_axes = {"images": {}, "keypoints": {}}
    if batch_size == 0:
        dynamic_axes["images"][0] = "batch_size"
        dynamic_axes["keypoints"][0] = "batch_size"
    if height == 0:
        dynamic_axes["images"][2] = "height"
    if width == 0:
        dynamic_axes["images"][3] = "width"
    dynamic_axes |= {"matches": {0: "num_matches"}, "mscores": {0: "num_matches"}}
    torch.onnx.export(
        pipeline,
        torch.zeros(batch_size or 2, extractor_type.input_channels, height or 256, width or 256),
        str(output),
        input_names=["images"],
        output_names=["keypoints", "matches", "mscores"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
    onnx.checker.check_model(output)
    onnx.save_model(SymbolicShapeInference.infer_shapes(onnx.load_model(output), auto_merge=True), output)  # type: ignore
    typer.echo(f"Successfully exported model to {output}")
    if fp16:
        typer.echo(
            "Converting to FP16. Warning: This FP16 model should NOT be used for TensorRT. TRT provides its own fp16 option."
        )
        onnx.save_model(convert_float_to_float16(onnx.load_model(output)), output.with_suffix(".fp16.onnx"))


@app.command()
def infer(
    model_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to ONNX model.")],
    left_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to first image.")
    ],
    right_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to second image.")
    ],
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output_path: Annotated[
        Optional[Path],  # noqa: UP007
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save output matches figure. If not given, show visualization.",
        ),
    ] = None,
    height: Annotated[
        int,
        typer.Option("-h", "--height", min=1, help="Height of input image at which to perform inference."),
    ] = 1024,
    width: Annotated[
        int,
        typer.Option("-w", "--width", min=1, help="Width of input image at which to perform inference."),
    ] = 1024,
    device: Annotated[
        InferenceDevice, typer.Option("-d", "--device", help="Device to run inference on.")
    ] = InferenceDevice.cpu,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether model uses FP16 precision.")] = False,
    profile: Annotated[bool, typer.Option("--profile", help="Whether to profile model execution.")] = False,
):
    """Run inference for LightGlue ONNX model."""
    import numpy as np
    import onnxruntime as ort

    from lightglue_dynamo import viz
    from lightglue_dynamo.preprocessors import DISKPreprocessor, SuperPointPreprocessor

    raw_images = [left_image_path, right_image_path]
    raw_images = [cv2.resize(cv2.imread(str(i)), (width, height)) for i in raw_images]
    images = np.stack(raw_images)
    match extractor_type:
        case Extractor.superpoint:
            images = SuperPointPreprocessor.preprocess(images)
        case Extractor.disk:
            images = DISKPreprocessor.preprocess(images)
    images = images.astype(np.float16 if fp16 and device != InferenceDevice.tensorrt else np.float32)

    session_options = ort.SessionOptions()
    session_options.enable_profiling = profile
    # session_options.optimized_model_filepath = "weights/ort_optimized.onnx"

    providers = [("CPUExecutionProvider", {})]
    if device == InferenceDevice.cuda:
        providers.insert(0, ("CUDAExecutionProvider", {}))
    elif device == InferenceDevice.tensorrt:
        providers.insert(0, ("CUDAExecutionProvider", {}))
        providers.insert(
            0,
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "weights/.trtcache_engines",
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": "weights/.trtcache_timings",
                    "trt_fp16_enable": fp16,
                },
            ),
        )
    elif device == InferenceDevice.openvino:
        providers.insert(0, ("OpenVINOExecutionProvider", {}))

    session = ort.InferenceSession(model_path, session_options, providers)

    for _ in range(100 if profile else 1):
        keypoints, matches, mscores = session.run(None, {"images": images})

    viz.plot_images(raw_images)
    viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
    if output_path is None:
        viz.plt.show()
    else:
        viz.save_plot(output_path)


@app.command()
def trtexec(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to ONNX model or built TensorRT engine."),
    ],
    left_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to first image.")
    ],
    right_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to second image.")
    ],
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output_path: Annotated[
        Optional[Path],  # noqa: UP007
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save output matches figure. If not given, show visualization.",
        ),
    ] = None,
    height: Annotated[
        int,
        typer.Option("-h", "--height", min=1, help="Height of input image at which to perform inference."),
    ] = 1024,
    width: Annotated[
        int,
        typer.Option("-w", "--width", min=1, help="Width of input image at which to perform inference."),
    ] = 1024,
    tf32: Annotated[bool, typer.Option("--tf32", help="Whether model uses TF32 precision.")] = False,
    bf16: Annotated[bool, typer.Option("--bf16", help="Whether model uses BF16 precision.")] = False,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether model uses FP16 precision.")] = False,
    fp8: Annotated[bool, typer.Option("--fp8", help="Whether model uses FP8 precision.")] = False,
    int8: Annotated[bool, typer.Option("--int8", help="Whether model uses INT8 precision.")] = False,
    calib_cache: Annotated[Optional[Path], typer.Option("--calib-cache", help="Path to calibration cache for INT8.")] = None,
    calib_image_dir: Annotated[Optional[Path], typer.Option("--calib-image-dir", exists=True, file_okay=False, help="Path to INT8 calibration images.")] = None,
    profile: Annotated[bool, typer.Option("--profile", help="Whether to profile model execution.")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Print debug information.")] = False,
    use_dla: Annotated[bool, typer.Option("--use-dla", help="Use DLA for inference if available.")] = False,
    allow_gpu_fallback: Annotated[bool, typer.Option("--allow-gpu-fallback", help="Use GPU fallback for DLA.")] = True,
):
    """Run pure TensorRT inference for LightGlue model using Polygraphy (requires TensorRT to be installed)."""
    import numpy as np
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import (
        Calibrator,
        CreateConfig,
        EngineFromBytes,
        EngineFromNetwork,
        NetworkFromOnnxPath,
        SaveEngine,
        TrtRunner,
    )

    from lightglue_dynamo import viz

    raw_images = [left_image_path, right_image_path]
    raw_images = [cv2.resize(cv2.imread(str(i)), (width, height)) for i in raw_images]
    images = np.stack(raw_images)

    print(f"Raw images shape: {images.shape}")
    print(f"Raw images dtype: {images.dtype}")
    print(f"Raw images range: [{images.min()}, {images.max()}]")

    match extractor_type:
        case Extractor.superpoint:
            images = SuperPointPreprocessor.preprocess(images)
        case Extractor.disk:
            images = DISKPreprocessor.preprocess(images)
    images = images.astype(np.float32)
    print(f"Preprocessed images shape: {images.shape}")
    print(f"Preprocessed images dtype: {images.dtype}")
    print(f"Preprocessed images range: [{images.min()}, {images.max()}]")

    print("Python input tensor debug:")
    for batch_idx in range(images.shape[0]):
        print(f"Batch {batch_idx} first 10 values:")
        flat_batch = images[batch_idx].flatten()
        for i in range(min(10, len(flat_batch))):
            print(f"  [{i}] = {flat_batch[i]:.6f}")
        
        print(f"Batch {batch_idx}: min={flat_batch.min():.6f}, max={flat_batch.max():.6f}")

    # Build TensorRT engine
    if model_path.suffix == ".engine":
        build_engine = EngineFromBytes(BytesFromPath(str(model_path)))
    else:  # .onnx
        calibrator = None
        if int8:
            if calib_image_dir is None:
                raise typer.Exit("Error: --calib-image-dir must be provided for INT8 quantization.")
            
            # Create the data loader for calibration
            data_loader = calib_data_loader(calib_image_dir, height, width, extractor_type)
            
            # Use the data loader with the calibrator
            calibrator = Calibrator(
                data_loader=data_loader,
                cache=str(calib_cache) if calib_cache else None
            )

        build_engine = EngineFromNetwork(
            NetworkFromOnnxPath(str(model_path)), 
            config=CreateConfig(
                tf32=tf32,
                fp16=fp16,
                bf16=bf16, 
                fp8=fp8,
                int8=int8,
                calibrator=calibrator,
                use_dla=use_dla, 
                allow_gpu_fallback=allow_gpu_fallback,
            )
        )
        engine_path = str(model_path.with_suffix(".engine"))
        build_engine = SaveEngine(build_engine, engine_path)
        print(f"TensorRT engine built and saved to: {engine_path}")

    with TrtRunner(build_engine) as runner:
        for _ in range(10 if profile else 1):  # Warm-up if profiling
            outputs = runner.infer(feed_dict={"images": images})
            keypoints, matches, mscores = outputs["keypoints"], outputs["matches"], outputs["mscores"]  # noqa: F841

        if profile:
            typer.echo(f"Inference Time: {runner.last_inference_time():.3f} s")

    if debug:
        print("\n=== PYTHON OUTPUT DEBUG ===")
        print(f"Keypoints shape: {keypoints.shape}")
        print(f"Matches shape: {matches.shape}")
        print(f"MScores shape: {mscores.shape}")

        print(f"Keypoints dtype: {keypoints.dtype}")
        print(f"Keypoints range: [{keypoints.min():.6f}, {keypoints.max():.6f}]")

        # Print first 20 keypoints for each batch (same as C++)
        print("Python keypoints batch 0:")
        for i in range(20):
            if i < keypoints.shape[1]:  # Check bounds
                x = keypoints[0, i, 0]
                y = keypoints[0, i, 1]
                print(f"  [{i}] x={x:.2f}, y={y:.2f}")

        print("Python keypoints batch 1:")
        for i in range(20):
            if i < keypoints.shape[1]:  # Check bounds
                x = keypoints[1, i, 0]
                y = keypoints[1, i, 1]
                print(f"  [{i}] x={x:.2f}, y={y:.2f}")

        print(f"Total matches: {matches.shape[0]}")
        print(f"Match scores range: [{mscores.min():.6f}, {mscores.max():.6f}]")

    # Check if keypoints are all zeros
    batch0_nonzero = np.count_nonzero(keypoints[0])
    batch1_nonzero = np.count_nonzero(keypoints[1])
    print(f"Batch 0 non-zero keypoint values: {batch0_nonzero}")
    print(f"Batch 1 non-zero keypoint values: {batch1_nonzero}")

    viz.plot_images(raw_images)
    viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
    if output_path is None:
        viz.plt.show()
    else:
        viz.save_plot(output_path)

if __name__ == "__main__":
    app()
