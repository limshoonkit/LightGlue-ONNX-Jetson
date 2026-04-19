import argparse
from typing import List

import torch

from lightglue_onnx import DISK, LightGlue, LightGlueEnd2End, SuperPoint
from lightglue_onnx.end2end import normalize_keypoints
from lightglue_onnx.utils import load_image, rgb_to_grayscale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=512,
        required=False,
        help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the image to this value. Otherwise, please provide two integers (height width).",
    )
    parser.add_argument(
        "--extractor_type",
        type=str,
        default="superpoint",
        choices=["superpoint", "disk", "aliked"],
        required=False,
        help="Type of feature extractor. 'aliked' is supported in --matcher_only mode (no extractor ONNX is written).",
    )
    parser.add_argument(
        "--lightglue_weights",
        type=str,
        default=None,
        required=False,
        help="Local path to LightGlue .pth weights (skips the hub download).",
    )
    parser.add_argument(
        "--extractor_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the feature extractor ONNX model.",
    )
    parser.add_argument(
        "--lightglue_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the LightGlue ONNX model.",
    )
    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Whether to export an end-to-end pipeline instead of individual models.",
    )
    parser.add_argument(
        "--matcher_only",
        action="store_true",
        help="Export only the LightGlue matcher ONNX; skip writing the extractor ONNX.",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Whether to allow dynamic image sizes."
    )

    # Extractor-specific args:
    parser.add_argument(
        "--max_num_keypoints",
        type=int,
        default=None,
        required=False,
        help="Maximum number of keypoints outputted by the extractor.",
    )

    return parser.parse_args()


def export_onnx(
    img_size=512,
    extractor_type="superpoint",
    extractor_path=None,
    lightglue_path=None,
    lightglue_weights=None,
    img0_path="assets/sacre_coeur1.jpg",
    img1_path="assets/sacre_coeur2.jpg",
    end2end=False,
    matcher_only=False,
    dynamic=False,
    max_num_keypoints=None,
):
    # Handle args
    if isinstance(img_size, List) and len(img_size) == 1:
        img_size = img_size[0]

    if end2end and matcher_only:
        raise ValueError("--end2end and --matcher_only are mutually exclusive.")
    if extractor_path is not None and end2end:
        raise ValueError(
            "Extractor will be combined with LightGlue when exporting end-to-end model."
        )
    if extractor_path is None:
        extractor_path = f"weights/{extractor_type}.onnx"
        if max_num_keypoints is not None:
            extractor_path = extractor_path.replace(
                ".onnx", f"_{max_num_keypoints}.onnx"
            )

    if lightglue_path is None:
        lightglue_path = (
            f"weights/{extractor_type}_lightglue"
            f"{'_end2end' if end2end else ''}"
            ".onnx"
        )

    extractor_type = extractor_type.lower()

    # ALIKED: matcher_only path — the legacy lightglue_onnx has no ALIKED extractor,
    # so we trace with dummy feats instead of running an extractor.
    if extractor_type == "aliked":
        if not matcher_only:
            raise NotImplementedError(
                "ALIKED export is only supported with --matcher_only (use the existing ALIKED ONNX extractor separately)."
            )
        lightglue = LightGlue(extractor_type, weights_path=lightglue_weights).eval()
        n = max_num_keypoints or 512
        kpts0 = torch.rand(1, n, 2) * 2 - 1
        kpts1 = torch.rand(1, n, 2) * 2 - 1
        desc0 = torch.randn(1, n, 128)
        desc1 = torch.randn(1, n, 128)
        matcher_axes: dict = {
            "matches0": {0: "num_matches0"},
            "mscores0": {0: "num_matches0"},
        }
        if dynamic:
            matcher_axes.update({
                "kpts0": {1: "num_keypoints0"},
                "kpts1": {1: "num_keypoints1"},
                "desc0": {1: "num_keypoints0"},
                "desc1": {1: "num_keypoints1"},
            })
        else:
            print(f"Fixing matcher num_keypoints to N={n} (pass --dynamic to keep it variable).")
        torch.onnx.export(
            lightglue,
            (kpts0, kpts1, desc0, desc1),
            lightglue_path,
            input_names=["kpts0", "kpts1", "desc0", "desc1"],
            output_names=["matches0", "mscores0"],
            opset_version=17,
            dynamic_axes=matcher_axes,
        )
        print(f"Exported standalone LightGlue matcher (aliked) to {lightglue_path}")
        return

    # Sample images for tracing
    image0, scales0 = load_image(img0_path, resize=img_size)
    image1, scales1 = load_image(img1_path, resize=img_size)
    # Models
    if extractor_type == "superpoint":
        # SuperPoint works on grayscale images.
        image0 = rgb_to_grayscale(image0)
        image1 = rgb_to_grayscale(image1)
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval()
        lightglue = LightGlue(extractor_type, weights_path=lightglue_weights).eval()
    elif extractor_type == "disk":
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval()
        lightglue = LightGlue(extractor_type, weights_path=lightglue_weights).eval()
    else:
        raise NotImplementedError(
            f"LightGlue has not been trained on {extractor_type} features."
        )

    # ONNX Export
    if end2end:
        pipeline = LightGlueEnd2End(extractor, lightglue).eval()

        dynamic_axes = {
            "kpts0": {1: "num_keypoints0"},
            "kpts1": {1: "num_keypoints1"},
            "matches0": {0: "num_matches0"},
            "mscores0": {0: "num_matches0"},
        }
        if dynamic:
            dynamic_axes.update(
                {
                    "image0": {2: "height0", 3: "width0"},
                    "image1": {2: "height1", 3: "width1"},
                }
            )

        torch.onnx.export(
            pipeline,
            (image0[None], image1[None]),
            lightglue_path,
            input_names=["image0", "image1"],
            output_names=[
                "kpts0",
                "kpts1",
                "matches0",
                "mscores0",
            ],
            opset_version=17,
            dynamic_axes=dynamic_axes,
        )
    else:
        # Export Extractor (unless matcher_only)
        if not matcher_only:
            dynamic_axes = {
                "keypoints": {1: "num_keypoints"},
                "scores": {1: "num_keypoints"},
                "descriptors": {1: "num_keypoints"},
            }
            if dynamic:
                dynamic_axes.update({"image": {2: "height", 3: "width"}})
            else:
                print(
                    f"WARNING: Exporting without --dynamic implies that the {extractor_type} extractor's input image size will be locked to {image0.shape[-2:]}"
                )
                extractor_path = extractor_path.replace(
                    ".onnx", f"_{image0.shape[-2]}x{image0.shape[-1]}.onnx"
                )

            torch.onnx.export(
                extractor,
                image0[None],
                extractor_path,
                input_names=["image"],
                output_names=["keypoints", "scores", "descriptors"],
                opset_version=17,
                dynamic_axes=dynamic_axes,
            )

        # Export LightGlue
        feats0, feats1 = extractor(image0[None]), extractor(image1[None])
        kpts0, scores0, desc0 = feats0
        kpts1, scores1, desc1 = feats1

        kpts0 = normalize_keypoints(kpts0, image0.shape[1], image0.shape[2])
        kpts1 = normalize_keypoints(kpts1, image1.shape[1], image1.shape[2])

        matcher_axes: dict = {
            "matches0": {0: "num_matches0"},
            "mscores0": {0: "num_matches0"},
        }
        if dynamic:
            matcher_axes.update({
                "kpts0": {1: "num_keypoints0"},
                "kpts1": {1: "num_keypoints1"},
                "desc0": {1: "num_keypoints0"},
                "desc1": {1: "num_keypoints1"},
            })
        else:
            print(f"Fixing matcher num_keypoints to traced N={kpts0.shape[1]} (pass --dynamic to keep it variable).")
        torch.onnx.export(
            lightglue,
            (kpts0, kpts1, desc0, desc1),
            lightglue_path,
            input_names=["kpts0", "kpts1", "desc0", "desc1"],
            output_names=["matches0", "mscores0"],
            opset_version=17,
            dynamic_axes=matcher_axes,
        )


if __name__ == "__main__":
    args = parse_args()
    export_onnx(**vars(args))
