"""Chain an extractor TRT engine with a LightGlue matcher TRT engine and visualize matches.

Assumes:
  - Extractor engine was built from a b=2 ONNX; outputs keys:
      keypoints (2, N, 2) in pixel coords, descriptors (2, N, D),
      num_keypoints (2,), keypoint_scores (2, N).
  - Matcher engine is the classic 4-input ONNX from export.py --matcher_only:
      inputs kpts0/1 (1, N, 2) normalized isotropically, desc0/1 (1, N, D);
      outputs matches0 (M, 2), mscores0 (M,).
"""
from pathlib import Path
import typer
import cv2
import numpy as np
from typing import Annotated, Optional

from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

from lightglue_dynamo.config import Extractor
from lightglue_dynamo.preprocessors import ALIKEDPreprocessor, SuperPointOpenPreprocessor
from lightglue_dynamo import viz


def normalize_keypoints_isotropic(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    """Match lightglue_onnx/end2end.py::normalize_keypoints: shift by center, scale by longer side/2."""
    size = np.array([w, h], dtype=np.float32)
    shift = size / 2
    scale = size.max() / 2
    return ((kpts.astype(np.float32) - shift) / scale).astype(np.float32)


def main(
    extractor_engine: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    matcher_engine: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    extractor_type: Annotated[Extractor, typer.Option()],
    img1: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    img2: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    height: Annotated[int, typer.Option("-h", "--height", min=1)],
    width: Annotated[int, typer.Option("-w", "--width", min=1)],
    output: Annotated[Optional[Path], typer.Option("-o", "--output", dir_okay=False)] = None,
):
    raw_images = [cv2.resize(cv2.imread(str(p)), (width, height)) for p in (img1, img2)]
    images = np.stack(raw_images)

    if extractor_type == Extractor.superpoint_open:
        images = SuperPointOpenPreprocessor.preprocess(images)
    elif extractor_type in (Extractor.aliked_n16, Extractor.aliked_n16rot, Extractor.aliked_n32, Extractor.aliked_t16):
        images = ALIKEDPreprocessor.preprocess(images)
    else:
        raise typer.BadParameter(f"Unsupported extractor {extractor_type} for this chained-matching script.")
    images = images.astype(np.float32)

    typer.echo(f"Running extractor engine {extractor_engine.name}...")
    ext_engine = EngineFromBytes(BytesFromPath(str(extractor_engine)))
    with TrtRunner(ext_engine) as runner:
        ext_out = runner.infer({"images": images})

    kpts_out = ext_out["keypoints"]           # (2, N, 2)
    descs = ext_out["descriptors"]            # (2, N, D)
    num_kpts = ext_out["num_keypoints"]       # (2,)
    n0, n1 = int(num_kpts[0]), int(num_kpts[1])

    # ALIKED emits anisotropic-normalized coords in [-1, 1] (see aliked/soft_detect.py).
    # SuperPoint-Open emits pixel coords. Convert ALIKED back to pixels so the downstream
    # isotropic normalization + plotting both see a consistent pixel-space input.
    if extractor_type in (Extractor.aliked_n16, Extractor.aliked_n16rot, Extractor.aliked_n32, Extractor.aliked_t16):
        size = np.array([width, height], dtype=np.float32)
        kpts_px = (kpts_out.astype(np.float32) + 1.0) * size / 2.0
    else:
        kpts_px = kpts_out.astype(np.float32)

    typer.echo(f"Extractor kpts: {kpts_px.shape}, desc: {descs.shape}, num_kpts: {(n0, n1)}")

    kpts0_n = normalize_keypoints_isotropic(kpts_px[0], height, width)[None]  # (1, N, 2)
    kpts1_n = normalize_keypoints_isotropic(kpts_px[1], height, width)[None]
    desc0 = descs[0:1].astype(np.float32)
    desc1 = descs[1:2].astype(np.float32)

    typer.echo(f"Running matcher engine {matcher_engine.name}...")
    match_engine = EngineFromBytes(BytesFromPath(str(matcher_engine)))
    with TrtRunner(match_engine) as runner:
        m_out = runner.infer({"kpts0": kpts0_n, "kpts1": kpts1_n, "desc0": desc0, "desc1": desc1})

    matches0 = m_out["matches0"]   # (M, 2)
    mscores0 = m_out["mscores0"]   # (M,)
    typer.echo(f"Matcher returned {matches0.shape[0]} raw matches.")

    # Drop any match whose index falls in the zero-padded tail of either image.
    valid = (matches0[:, 0] < n0) & (matches0[:, 1] < n1)
    matches0 = matches0[valid]
    mscores0 = mscores0[valid]
    typer.echo(f"{matches0.shape[0]} matches after padding filter.")

    viz.plot_images(raw_images)
    viz.plot_matches(kpts_px[0][matches0[:, 0]], kpts_px[1][matches0[:, 1]], color="lime", lw=0.2)

    if output is None:
        viz.plt.show()
    else:
        viz.save_plot(output)
        typer.echo(f"Saved visualization to {output}")


if __name__ == "__main__":
    typer.run(main)
