#!/usr/bin/env bash
# Export + build extractor and standalone LightGlue matcher TRT engines, then visualize.
#
# For each target (extractor, batch, height, width, kp):
#   1. Export extractor ONNX (if missing) via `dynamo.py export`
#   2. Simplify + build extractor TRT engine
#   3. Export standalone LightGlue matcher ONNX (if missing) via `export.py --matcher_only`
#   4. Simplify + build matcher TRT engine
#   5. If batch=2, run chained extractor -> matcher inference and save a match-viz PNG
#
# Filename convention:
#   extractor : <extractor>_b<B>_h<H>_w<W>_kp<K>.onnx
#               e.g. aliked_n16rot_b2_h480_w768_kp256.onnx
#   matcher   : <variant>_lightglue_matcher_kp<K>.onnx
#               variant = superpoint (for superpoint_open) or aliked (for aliked_*)
#
# Usage:
#   ./build_engines.sh [img1] [img2]
#   PRECISION_FLAGS="--fp16" ./build_engines.sh   # override trt precision

set -euo pipefail

WEIGHTS_DIR="weights/euroc"
IMG1="${1:-assets/DSC_0410.JPG}"
IMG2="${2:-assets/DSC_0411.JPG}"
PRECISION_FLAGS="${PRECISION_FLAGS:---fp16}"

# Resolve trtexec: env var override > $PATH > common install locations.
if [[ -z "${TRTEXEC:-}" ]]; then
    if command -v trtexec >/dev/null 2>&1; then
        TRTEXEC="$(command -v trtexec)"
    else
        for cand in \
            /usr/src/tensorrt/bin/trtexec \
            /home/ubuntu/Third_party/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/bin/trtexec \
            /home/ubuntu/Third_party/TensorRT/build/out/trtexec; do
            [[ -x "$cand" ]] && TRTEXEC="$cand" && break
        done
    fi
fi
if [[ -z "${TRTEXEC:-}" || ! -x "$TRTEXEC" ]]; then
    echo "ERROR: trtexec not found. Set TRTEXEC=/path/to/trtexec or add it to PATH." >&2
    exit 1
fi
echo "Using trtexec: $TRTEXEC"

# (extractor, batch, height, width, kp) tuples
TARGETS=(
    "superpoint_open 2 480 768 128"
    "superpoint_open 2 480 768 256"
    "aliked_n16rot   2 480 768 128"
    "aliked_n16rot   2 480 768 256"
)

matcher_variant_for() {
    # stdout: "superpoint" | "aliked" | ""
    case "$1" in
        superpoint_open)                                                          echo "superpoint" ;;
        aliked_n16|aliked_n16rot|aliked_n32|aliked_t16)                           echo "aliked" ;;
        *)                                                                        echo "" ;;
    esac
}

mkdir -p "$WEIGHTS_DIR"

for target in "${TARGETS[@]}"; do
    read -r extractor batch height width kp <<< "$target"

    base="${extractor}_b${batch}_h${height}_w${width}_kp${kp}"
    onnx="$WEIGHTS_DIR/${base}.onnx"
    simplified="$WEIGHTS_DIR/${base}_simplify.onnx"
    engine="$WEIGHTS_DIR/${base}_simplify.engine"

    if [[ "$batch" == "1" ]]; then
        trt_images=("$IMG1")
    else
        trt_images=("$IMG1" "$IMG2")
    fi

    echo ""
    echo "=============================================================="
    echo "  EXTRACTOR : $extractor   b=$batch  h=$height  w=$width  kp=$kp"
    echo "=============================================================="

    # --- Extractor: export (if missing) + simplify + build engine ---
    if [[ -f "$onnx" ]]; then
        echo "[skip export] $onnx already exists"
    else
        echo "  [export extractor] $onnx"
        python dynamo.py export "$extractor" \
            -o "$onnx" \
            -b "$batch" -h "$height" -w "$width" \
            --num-keypoints "$kp"
    fi

    if [[ -f "$simplified" ]]; then
        echo "[skip onnxsim] $simplified already exists"
    else
        onnxsim "$onnx" "$simplified"
    fi

    if [[ -f "$engine" ]]; then
        echo "[skip trtexec] $engine already exists"
    else
        python dynamo.py trtexec \
            "$simplified" \
            "${trt_images[@]}" \
            -e "$extractor" \
            -h "$height" -w "$width" \
            $PRECISION_FLAGS \
            --profile
    fi

    # --- Matcher: pair by variant + kp ---
    variant="$(matcher_variant_for "$extractor")"
    if [[ -z "$variant" ]]; then
        echo "[no matcher] $extractor has no standalone LightGlue variant; skipping matcher stage"
        continue
    fi

    matcher_onnx="$WEIGHTS_DIR/${variant}_lightglue_matcher_kp${kp}.onnx"
    matcher_simplified="$WEIGHTS_DIR/${variant}_lightglue_matcher_kp${kp}_simplify.onnx"
    matcher_engine="$WEIGHTS_DIR/${variant}_lightglue_matcher_kp${kp}_simplify.engine"

    if [[ ! -f "$matcher_onnx" ]]; then
        echo ""
        echo "  MATCHER    : exporting $matcher_onnx  (variant=$variant kp=$kp)"
        python export.py \
            --extractor_type "$variant" \
            --matcher_only \
            --lightglue_weights "weights/${variant}_lightglue.pth" \
            --max_num_keypoints "$kp" \
            --lightglue_path "$matcher_onnx"
    else
        echo "[skip export] $matcher_onnx already exists"
    fi

    if [[ -f "$matcher_simplified" ]]; then
        echo "[skip onnxsim] $matcher_simplified already exists"
    else
        onnxsim "$matcher_onnx" "$matcher_simplified"
    fi

    if [[ -f "$matcher_engine" ]]; then
        echo "[skip trtexec] $matcher_engine already exists"
    else
        "$TRTEXEC" \
            --onnx="$matcher_simplified" \
            --saveEngine="$matcher_engine" \
            $PRECISION_FLAGS
    fi

    # --- Always visualize when batch=2 ---
    if [[ "$batch" == "2" ]]; then
        viz_out="$WEIGHTS_DIR/${base}_matches.png"
        echo ""
        echo "  CHAINED MATCH -> $viz_out"
        python match_engines.py \
            --extractor-engine "$engine" \
            --matcher-engine "$matcher_engine" \
            --extractor-type "$extractor" \
            --img1 "$IMG1" --img2 "$IMG2" \
            -h "$height" -w "$width" \
            -o "$viz_out"
    else
        echo "[skip match] batch=$batch (visualization requires b=2)"
    fi
done

echo ""
echo "Done."
