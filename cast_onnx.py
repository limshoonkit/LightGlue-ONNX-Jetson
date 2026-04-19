import onnx
import onnx_graphsurgeon as gs
import numpy as np

# === Input & output paths ===
onnx_in  = "weights/superpoint_lightglue_b2_h400_w640_kp256.onnx"
onnx_out = "weights/superpoint_lightglue_b2_h400_w640_kp256_int32.onnx"

# === Load graph ===
graph = gs.import_onnx(onnx.load(onnx_in))

# === Identify outputs that need casting ===
targets = {"keypoints", "matches"}  # names to cast

new_outputs = []
for out in graph.outputs:
    if out.name in targets and out.dtype == np.int64:
        print(f"[INFO] Casting {out.name} from int64 -> int32")

        # Create a Cast node
        cast_out = gs.Variable(out.name + "_int32", dtype=np.int32, shape=out.shape)
        cast_node = gs.Node(op="Cast", inputs=[out], outputs=[cast_out], attrs={"to": onnx.TensorProto.INT32})

        # Replace graph output with casted version
        graph.nodes.append(cast_node)
        new_outputs.append(cast_out)
    else:
        new_outputs.append(out)

graph.outputs = new_outputs

# === Cleanup and save ===
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnx_out)

print(f"[OK] Patched model saved to {onnx_out}")
