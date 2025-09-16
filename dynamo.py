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
from lightglue_dynamo.preprocessors import DISKPreprocessor, SuperPointPreprocessor, SuperPointOpenPreprocessor, ALIKEDPreprocessor

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
            case Extractor.superpoint_open:
                images = SuperPointOpenPreprocessor.preprocess(images)
            case Extractor.disk:
                images = DISKPreprocessor.preprocess(images)
            case Extractor.aliked_n16 | Extractor.aliked_n16rot | Extractor.aliked_n32 | Extractor.aliked_t16:
                images = ALIKEDPreprocessor.preprocess(images)
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
        Optional[Path],
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
    opset: Annotated[int, typer.Option(min=16, max=20, help="ONNX opset version of exported model.")] = 17,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether to also convert to FP16.")] = False,
):
    """Export LightGlue to ONNX."""
    import onnx
    import torch
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    from onnxruntime.transformers.float16 import convert_float_to_float16

    from lightglue_dynamo.models import DISK, LightGlue, Pipeline, SuperPoint, SuperPointOpen, ALIKED
    from lightglue_dynamo.ops import use_fused_multi_head_attention

    # A simple helper for type checking
    def check_multiple_of(value: int, divisor: int):
        if value > 0 and value % divisor != 0:
            msg = f"Value {value} must be a multiple of {divisor}."
            raise typer.BadParameter(msg)

    match extractor_type:
        case Extractor.superpoint:
            extractor = SuperPoint(num_keypoints=num_keypoints)
        case Extractor.superpoint_open:
            extractor = SuperPointOpen(detection_threshold=0.005, nms_radius=5, max_num_keypoints=num_keypoints)
        case Extractor.disk:
            extractor = DISK(num_keypoints=num_keypoints)
        case Extractor.aliked_n16:
            model_name = "aliked-n16"
            weights_path = Path(f"./weights/{model_name}.pth")
            extractor = ALIKED(model_name=model_name,
                            device="cpu", 
                            n_limit=num_keypoints, 
                            pretrained_path=str(weights_path))
        case Extractor.aliked_n16rot:
            model_name = "aliked-n16rot"
            weights_path = Path(f"./weights/{model_name}.pth")
            extractor = ALIKED(model_name=model_name,
                            device="cpu",
                            n_limit=num_keypoints, 
                            pretrained_path=str(weights_path))
        case Extractor.aliked_n32:
            model_name = "aliked-n32"
            weights_path = Path(f"./weights/{model_name}.pth")
            extractor = ALIKED(model_name=model_name, 
                            device="cpu", 
                            n_limit=num_keypoints, 
                            pretrained_path=str(weights_path))
        case Extractor.aliked_t16:
            model_name = "aliked-t16"
            weights_path = Path(f"./weights/{model_name}.pth")
            extractor = ALIKED(model_name=model_name, 
                            device="cpu", 
                            n_limit=num_keypoints, 
                            pretrained_path=str(weights_path))

    check_multiple_of(height, extractor_type.input_dim_divisor)
    check_multiple_of(width, extractor_type.input_dim_divisor)

    if height > 0 and width > 0 and num_keypoints > height * width:
        raise typer.BadParameter("num_keypoints cannot be greater than height * width.")

    dummy_input = torch.randn(batch_size or 2, extractor_type.input_channels, height or 256, width or 256)
    
    # Handle SuperPoint Open special case
    if extractor_type == Extractor.superpoint_open:
        typer.echo(f"Exporting {extractor_type} extractor...")
        ckpt = torch.load("./weights/superpoint_v6_from_tf.pth", map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        
        # Load weights into the extractor
        extractor.load_state_dict(ckpt, strict=True)
        
        # Set to eval mode and disable gradients
        extractor.eval()
        for param in extractor.parameters():
            param.requires_grad = False
        
        # Set deterministic behavior
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        class ExtractorWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.model.eval()

            def forward(self, image):
                self.model.eval()
                preds = self.model({"image": image})
                return (
                    preds["keypoints"],
                    preds["keypoint_scores"],
                    preds["descriptors"],
                    preds["num_keypoints"],
                )

        model_to_export = ExtractorWrapper(extractor)
        model_to_export.eval()
        
        dummy_input = torch.randn(batch_size or 2, 1, height or 400, width or 640, dtype=torch.float32)
        
        output_names = ["keypoints", "keypoint_scores", "descriptors", "num_keypoints"]
        
        dynamic_axes = {
            "images": {},
            "keypoints": {},
            "keypoint_scores": {},
            "descriptors": {},
            "num_keypoints": {},
        }
        
        if batch_size == 0:
            dynamic_axes["images"][0] = "batch_size"
            dynamic_axes["keypoints"][0] = "batch_size"
            dynamic_axes["keypoint_scores"][0] = "batch_size"
            dynamic_axes["descriptors"][0] = "batch_size"
            dynamic_axes["num_keypoints"][0] = "batch_size"

        # Export with very specific settings
        torch.onnx.export(
            model_to_export,
            dummy_input,
            str(output),
            input_names=["images"],
            output_names=output_names,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )
    
    # Handle ALIKED extractor-only export
    # https://github.com/fabio-sim/LightGlue-ONNX/issues/69#issuecomment-2972612844
    elif extractor_type in [Extractor.aliked_n16, Extractor.aliked_n16rot, Extractor.aliked_n32, Extractor.aliked_t16]:
        typer.echo(f"Exporting {extractor_type} extractor...")
        
        from torch.onnx import register_custom_op_symbolic
        from torch.onnx.symbolic_helper import _get_const

        def deform_conv2d_symbolic(g, input, weight, offset, mask, bias,
                                   stride_h, stride_w, pad_h, pad_w,
                                   dil_h, dil_w, n_weight_grps,
                                   n_offset_grps, use_mask):
            
            stride = [_get_const(stride_h, 'i', 'stride_h'), _get_const(stride_w, 'i', 'stride_w')]
            padding = [_get_const(pad_h, 'i', 'pad_h'), _get_const(pad_w, 'i', 'pad_w')]
            dilation = [_get_const(dil_h, 'i', 'dil_h'), _get_const(dil_w, 'i', 'dil_w')]
            groups = _get_const(n_weight_grps, 'i', 'n_weight_grps')
            offset_groups = _get_const(n_offset_grps, 'i', 'n_offset_grps')
            
            if mask.node().mustBeNone():
                mask = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
            if bias.node().mustBeNone():
                bias = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

            return g.op("DeformConv", input, weight, offset, mask, bias,
                        stride_i=stride,
                        padding_i=padding,
                        dilation_i=dilation,
                        group_i=groups,
                        offset_group_i=offset_groups)

        # Register symbolic function for the 'torchvision::deform_conv2d' op
        # Hardcode opset 19, where DeformConv is a standard operator.
        opset_version_for_aliked = 19
        register_custom_op_symbolic("torchvision::deform_conv2d", deform_conv2d_symbolic, opset_version_for_aliked)

        class ALIKEDWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.model.eval()

            def forward(self, image):
                preds = self.model(image)
                keypoints = preds["keypoints"][0]  # Shape: [N, 2]
                scores = preds["scores"][0]          # Shape: [N]
                descriptors = preds["descriptors"][0] # Shape: [N, D]
                keypoints = keypoints.unsqueeze(0)
                scores = scores.unsqueeze(0)
                descriptors = descriptors.unsqueeze(0)

                # ALIKED outputs keypoints in normalized [-1, 1] range. Convert to pixel coordinates.
                _, _, h, w = image.shape
                wh = torch.tensor([w - 1, h - 1], device=keypoints.device, dtype=keypoints.dtype)
                keypoints = wh * (keypoints + 1) / 2
                
                # Calculate the number of keypoints for each image in the batch
                num_keypoints = torch.tensor([keypoints.shape[1]] * keypoints.shape[0], device=keypoints.device)
                
                return (
                    keypoints,      # B x N x 2
                    scores,         # B x N
                    descriptors,    # B x N x D
                    num_keypoints   # B
                )
        
        # Instantiate and wrap the model
        model_to_export = ALIKEDWrapper(extractor).eval()
        
        # Prepare for export
        dummy_input = torch.randn(batch_size or 1, 3, height or 480, width or 640, dtype=torch.float32)
        
        output_names = ["keypoints", "keypoint_scores", "descriptors", "num_keypoints"]
        
        dynamic_axes = {
            "images": {},
            "keypoints": {"num_keypoints": 1},
            "keypoint_scores": {"num_keypoints": 1},
            "descriptors": {"num_keypoints": 1},
        }
        
        if batch_size == 0:
            dynamic_axes["images"][0] = "batch_size"
            dynamic_axes["keypoints"][0] = "batch_size"
            dynamic_axes["keypoint_scores"][0] = "batch_size"
            dynamic_axes["descriptors"][0] = "batch_size"
            dynamic_axes["num_keypoints"] = {0: "batch_size"}

        if height == 0:
            dynamic_axes["images"][2] = "height"
        if width == 0:
            dynamic_axes["images"][3] = "width"

        torch.onnx.export(
            model_to_export,
            (dummy_input,),
            str(output),
            input_names=["images"],
            output_names=output_names,
            opset_version=opset_version_for_aliked,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )
    
    # Handle full pipeline export (SuperPoint + DISK with LightGlue)
    else:
        typer.echo(f"Exporting {extractor_type} pipeline...")

        check_multiple_of(batch_size, 2)
        matcher = LightGlue(**extractor_type.lightglue_config)
        model_to_export = Pipeline(extractor, matcher).eval()

        if fuse_multi_head_attention:
            typer.echo(
                "Warning: MHA nodes will be fused. Exported model is ONNX Runtime-specific."
            )
            if torch.__version__ < "2.4":
                raise typer.Abort("Fused multi-head attention requires PyTorch 2.4 or later.")
            use_fused_multi_head_attention()

        dynamic_axes = {"images": {}}
        if batch_size == 0:
            dynamic_axes["images"][0] = "batch_size"
        if height == 0:
            dynamic_axes["images"][2] = "height"
        if width == 0:
            dynamic_axes["images"][3] = "width"

        output_names = ["keypoints", "matches", "mscores"]
        dynamic_axes["keypoints"] = {}
        if batch_size == 0:
            dynamic_axes["keypoints"][0] = "batch_size"
        dynamic_axes |= {"matches": {0: "num_matches"}, "mscores": {0: "num_matches"}}

        torch.onnx.export(
            model_to_export,
            dummy_input,
            str(output),
            input_names=["images"],
            output_names=output_names,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
        )

    onnx.checker.check_model(output)
    onnx.save_model(SymbolicShapeInference.infer_shapes(onnx.load_model(output), auto_merge=True), output)  # type: ignore
    typer.echo(f"Successfully exported model to {output}")

    if fp16:
        typer.echo(
            "Converting to FP16. Warning: Do NOT use this for TensorRT, which has its own FP16 mode."
        )
        fp16_output = output.with_suffix(".fp16.onnx")
        onnx.save_model(convert_float_to_float16(onnx.load_model(output)), fp16_output)
        typer.echo(f"Successfully converted model to {fp16_output}")


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
    from lightglue_dynamo.preprocessors import DISKPreprocessor, SuperPointPreprocessor, SuperPointOpenPreprocessor, ALIKEDPreprocessor

    raw_images = [left_image_path, right_image_path]
    raw_images = [cv2.resize(cv2.imread(str(i)), (width, height)) for i in raw_images]
    images = np.stack(raw_images)
    match extractor_type:
        case Extractor.superpoint:
            images = SuperPointPreprocessor.preprocess(images)
        case Extractor.superpoint_open:
            images = SuperPointOpenPreprocessor.preprocess(images)
        case Extractor.disk:
            images = DISKPreprocessor.preprocess(images)
        case Extractor.aliked_n16 | Extractor.aliked_n16rot | Extractor.aliked_n32 | Extractor.aliked_t16:
            images = ALIKEDPreprocessor.preprocess(images)
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
        outputs = session.run(None, {"images": images})

    # Handle different output formats
    if extractor_type == Extractor.superpoint_open:
        typer.echo("Visualizing keypoints from SuperPointOpen extractor.")
        kpts, scores, descriptors, num_kpts = outputs
        print(f"Keypoints: {kpts.shape}")
        print(f"Keypoint scores: {scores.shape}")
        print(f"Descriptors: {descriptors.shape}")
        print(f"Number of keypoints: {num_kpts.shape}")
        viz.plot_extractor_only(raw_images, images.shape[0], kpts, num_kpts)
    elif extractor_type in [Extractor.aliked_n16, Extractor.aliked_n16rot, Extractor.aliked_n32, Extractor.aliked_t16]:
        typer.echo("Visualizing keypoints from ALIKED extractor.")
        kpts, scores, descriptors, num_kpts = outputs
        print(f"Keypoints: {kpts.shape}")
        print(f"Keypoint scores: {scores.shape}")
        print(f"Descriptors: {descriptors.shape}")
        print(f"Number of keypoints: {num_kpts.shape}")
        viz.plot_extractor_only(raw_images, images.shape[0], kpts, num_kpts)  # Reuse same visualization
    else:
        # Full pipeline model: outputs are [keypoints, matches, mscores]
        viz.plot_images(raw_images)
        typer.echo("Visualizing matches from LightGlue pipeline.")
        keypoints, matches, mscores = outputs
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
            images = images.astype(np.float32)
        case Extractor.superpoint_open:
            print("Using SuperPointOpenPreprocessor")
            images = SuperPointOpenPreprocessor.preprocess(images)
        case Extractor.disk:
            images = DISKPreprocessor.preprocess(images)
            images = images.astype(np.float32)
        case Extractor.aliked_n16 | Extractor.aliked_n16rot | Extractor.aliked_n32 | Extractor.aliked_t16:
            print(f"Using ALIKEDPreprocessor for {extractor_type}")
            images = ALIKEDPreprocessor.preprocess(images)
            images = images.astype(np.float32)

    if(debug):
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
        for _ in range(20 if profile else 1):
            outputs = runner.infer(feed_dict={"images": images})
        if profile:
            typer.echo(f"Inference Time: {runner.last_inference_time():.3f} s")
    
    if extractor_type == Extractor.superpoint_open:
        # Extractor-only model: outputs dict with keys "keypoints", "keypoint_scores", "descriptors"
        typer.echo("Visualizing keypoints from SuperPointOpen extractor.")
        kpts = outputs['keypoints']
        scores = outputs['keypoint_scores']
        descriptors = outputs['descriptors']
        num_kpts = outputs['num_keypoints']

        print(f"Keypoints: {kpts.shape}")
        print(f"Keypoint scores: {scores.shape}")
        print(f"Descriptors: {descriptors.shape}")
        print(f"Number of keypoints: {num_kpts.shape}")
        viz.plot_extractor_only(raw_images, images.shape[0], kpts, num_kpts)

        if debug:
            print("\n=== PYTHON OUTPUT DEBUG (Extractor) ===")
            print(f"Keypoints shape: {keypoints.shape}")
    elif extractor_type in [Extractor.aliked_n16, Extractor.aliked_n16rot, Extractor.aliked_n32, Extractor.aliked_t16]:
        # ALIKED Extractor-only model: outputs dict with keys "keypoints", "keypoint_scores", "descriptors"
        typer.echo(f"Visualizing keypoints from {extractor_type} extractor.")
        kpts = outputs['keypoints']
        scores = outputs['keypoint_scores']
        descriptors = outputs['descriptors']
        num_kpts = outputs['num_keypoints']

        print(f"Keypoints: {kpts.shape}")
        print(f"Keypoint scores: {scores.shape}")
        print(f"Descriptors: {descriptors.shape}")
        print(f"Number of keypoints: {num_kpts.shape}")
        viz.plot_extractor_only(raw_images, images.shape[0], kpts, num_kpts)

        if debug:
            print("\n=== PYTHON OUTPUT DEBUG (ALIKED Extractor) ===")
            print(f"Keypoints shape: {kpts.shape}")
            print(f"Keypoint scores shape: {scores.shape}")
            print(f"Descriptors shape: {descriptors.shape}")
    else:
        # SPLG pipeline model: outputs dict with keys "keypoints", "matches", "mscores"
        viz.plot_images(raw_images)
        typer.echo("Visualizing matches from LightGlue pipeline.")
        keypoints, matches, mscores = outputs["keypoints"], outputs["matches"], outputs["mscores"]
        viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
        if debug:
            print("\n=== PYTHON OUTPUT DEBUG (Pipeline) ===")
            print(f"Keypoints shape: {keypoints.shape}")
            print(f"Matches shape: {matches.shape}")

    if output_path is None:
        viz.plt.show()
    else:
        viz.save_plot(output_path)

if __name__ == "__main__":
    app()