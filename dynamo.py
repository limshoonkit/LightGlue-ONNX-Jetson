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

    # Set deterministic behavior
    torch.manual_seed(42) # or 69
    torch.cuda.manual_seed_all(42) # or 69
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

        # Export with specific settings
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
        opset_version_for_aliked = 19

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#DeformConv
        def deform_conv2d_symbolic(g, input, weight, offset, mask, bias,
                        stride_h, stride_w, pad_h, pad_w,
                        dil_h, dil_w, n_weight_grps,
                        n_offset_grps, use_mask):

            stride = [_get_const(stride_h, 'i', 'stride_h'), _get_const(stride_w, 'i', 'stride_w')]
            padding = [_get_const(pad_h, 'i', 'pad_h'), _get_const(pad_w, 'i', 'pad_w')]
            dilations = [_get_const(dil_h, 'i', 'dil_h'), _get_const(dil_w, 'i', 'dil_w')]
            groups = _get_const(n_weight_grps, 'i', 'n_weight_grps')
            offset_groups = _get_const(n_offset_grps, 'i', 'n_offset_grps')

            # [y_start, x_start, y_end, x_end]
            pads = [padding[0], padding[1], padding[0], padding[1]] 

            if mask.node().mustBeNone():
                mask = g.op("Constant")
            if bias.node().mustBeNone():
                bias = g.op("Constant")

            return g.op("DeformConv", input, weight, offset, mask, bias,
                        strides_i=stride,
                        pads_i=pads,
                        dilations_i=dilations,
                        group_i=groups,
                        offset_group_i=offset_groups)

        # Register symbolic function for the 'torchvision::deform_conv2d' op
        register_custom_op_symbolic("torchvision::deform_conv2d", deform_conv2d_symbolic, opset_version_for_aliked)

        class AlikedWrapper(torch.nn.Module):
            """Wraps the ALIKED model to output fixed-size tensors"""
            def __init__(self, model, max_kps=256):
                super().__init__()
                self.model = model
                self.model.eval()
                self.max_kps = max_kps

            def forward(self, image):
                # image shape: [B, 3, H, W]
                preds = self.model.forward(image)
                
                kpts_list, scores_list, desc_list = [], [], []
                batch_size = image.shape[0]
                
                # Loop through each item in the batch to handle padding
                for i in range(batch_size):
                    kpts = preds['keypoints'][i]    # Shape: [N, 2]
                    scores = preds['scores'][i]      # Shape: [N]
                    descriptors = preds['descriptors'][i] # Shape: [N, D]
                    
                    num_detected = kpts.shape[0]
                    
                    if num_detected > self.max_kps:
                        print(f"Warning: DKD produced {num_detected} keypoints, limiting to {self.max_kps}")
                        # Sort by scores and take top max_kps keypoints
                        top_indices = torch.topk(scores, self.max_kps, sorted=False)[1]
                        kpts = kpts[top_indices]
                        scores = scores[top_indices]
                        descriptors = descriptors[top_indices]
                        num_detected = self.max_kps
                    
                    # Create padded tensors with a fixed size (max_kps)
                    kpts_padded = torch.zeros(self.max_kps, 2, device=kpts.device, dtype=kpts.dtype)
                    scores_padded = torch.zeros(self.max_kps, device=scores.device, dtype=scores.dtype)
                    desc_padded = torch.zeros(self.max_kps, descriptors.shape[1], device=descriptors.device, dtype=descriptors.dtype)
                    
                    # Copy the valid data into the padded tensors
                    if num_detected > 0:
                        kpts_padded[:num_detected] = kpts
                        scores_padded[:num_detected] = scores
                        desc_padded[:num_detected] = descriptors
                    
                    kpts_list.append(kpts_padded)
                    scores_list.append(scores_padded)
                    desc_list.append(desc_padded)

                # Stack the list of tensors into a single batch tensor
                final_kpts = torch.stack(kpts_list, dim=0)
                final_scores = torch.stack(scores_list, dim=0)
                final_descriptors = torch.stack(desc_list, dim=0)
                
                # Count actual valid keypoints per batch item
                num_valid_keypoints = torch.tensor(
                    [min(preds['keypoints'][i].shape[0], self.max_kps) for i in range(batch_size)], 
                    device=image.device,
                    dtype=torch.int32
                )

                # Return tensors with fixed dimension
                return final_kpts, final_scores, final_descriptors, num_valid_keypoints

        def configure_aliked_for_export(extractor, max_kps):
            """Configure the ALIKED extractor to limit keypoint detection."""
            # Configure the DKD module parameters
            if hasattr(extractor, 'dkd'):
                print(f"Original DKD config: top_k={extractor.dkd.top_k}, n_limit={extractor.dkd.n_limit}, scores_th={extractor.dkd.scores_th}")
                
                # Set n_limit to max_kps to hard limit the number of keypoints
                extractor.dkd.n_limit = max_kps
                
                # Use top_k mode instead of threshold mode for consistent output
                if extractor.dkd.top_k <= 0:  # Currently in threshold mode
                    extractor.dkd.top_k = max_kps
                else:
                    extractor.dkd.top_k = min(extractor.dkd.top_k, max_kps)
                
                # Optionally increase score threshold to get fewer detections
                extractor.dkd.scores_th = max(extractor.dkd.scores_th, 0.25)
                
                print(f"Updated DKD config: top_k={extractor.dkd.top_k}, n_limit={extractor.dkd.n_limit}, scores_th={extractor.dkd.scores_th}")
            
            return extractor

        # Configure the extractor before wrapping
        extractor = configure_aliked_for_export(extractor, num_keypoints)
        model_to_export = AlikedWrapper(extractor, max_kps=num_keypoints)
        model_to_export.eval()

        output_names = ["keypoints", "keypoint_scores", "descriptors", "num_keypoints"]

        dummy_input = torch.randn(batch_size or 2, 3, height or 400, width or 640, dtype=torch.float32)

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

        torch.onnx.export(
            model_to_export,
            dummy_input,
            str(output),
            input_names=["images"],
            output_names=output_names,
            opset_version=max(opset, opset_version_for_aliked), # Use opset 19+ for DeformConv
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
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

    model = onnx.load(output)
    model = onnx.shape_inference.infer_shapes(model)
    model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    onnx.save(model, str(output))
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
        viz.plot_extractor_only(raw_images, images.shape[0], kpts, num_kpts, extractor_name="SuperPoint-Open")
    elif extractor_type in [Extractor.aliked_n16, Extractor.aliked_n16rot, Extractor.aliked_n32, Extractor.aliked_t16]:
        typer.echo("Visualizing keypoints from ALIKED extractor.")
        kpts, scores, descriptors, num_kpts = outputs
        print(f"Keypoints: {kpts.shape}")
        print(f"Keypoint scores: {scores.shape}")
        print(f"Descriptors: {descriptors.shape}")
        print(f"Number of keypoints: {num_kpts.shape}")
        viz.plot_extractor_only(raw_images, images.shape[0], kpts, num_kpts, extractor_name="ALIKED")
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
        viz.plot_extractor_only(raw_images, images.shape[0], kpts, num_kpts, extractor_name="SuperPoint-Open")

        if debug:
            print("\n=== PYTHON OUTPUT DEBUG (Extractor) ===")
            print(f"Keypoints: {kpts}")
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
        viz.plot_extractor_only(raw_images, images.shape[0], kpts, num_kpts, extractor_name="ALIKED")
        if debug:
            print("\n=== PYTHON OUTPUT DEBUG (Extractor) ===")
            print(f"Keypoints: {kpts}")
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