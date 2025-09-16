from enum import Enum
from typing import Any


class InferenceDevice(str, Enum):
    # Explicitly assign the string value instead of using auto()
    cpu = "cpu"
    cuda = "cuda"
    tensorrt = "tensorrt"
    openvino = "openvino"

    def __str__(self) -> str:
        """Return the string value of the enum member."""
        return self.value


class Extractor(str, Enum):
    # Explicitly assign the string value instead of using auto()
    superpoint = "superpoint"
    superpoint_open = "superpoint_open"
    disk = "disk"
    aliked = "aliked"
    aliked_n16 = "aliked_n16"
    aliked_n16rot = "aliked_n16rot"
    aliked_n32 = "aliked_n32"
    aliked_t16 = "aliked_t16"

    def __str__(self) -> str:
        """Return the string value of the enum member."""
        return self.value

    @property
    def input_dim_divisor(self) -> int:
        match self:
            case Extractor.superpoint:
                return 8
            case Extractor.superpoint_open:
                return 8
            case Extractor.disk:
                return 16
            case Extractor.aliked | Extractor.aliked_n16 | Extractor.aliked_n16rot | Extractor.aliked_n32 | Extractor.aliked_t16:
                return 32

    @property
    def input_channels(self) -> int:
        match self:
            case Extractor.superpoint:
                return 1
            case Extractor.superpoint_open:
                return 1
            case Extractor.disk:
                return 3
            case Extractor.aliked | Extractor.aliked_n16 | Extractor.aliked_n16rot | Extractor.aliked_n32 | Extractor.aliked_t16:
                return 3

    @property
    def lightglue_config(self) -> dict[str, Any]:
        match self:
            case Extractor.superpoint:
                return {"url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"}
            case Extractor.superpoint_open:
                return {"url": "./weights/superpoint_v6_from_tf.pth"}
            case Extractor.aliked_n16:
                return {"url": "./weights/aliked-n16.pth"}
            case Extractor.aliked_n16rot:
                return {"url": "./weights/aliked-n16rot.pth"}
            case Extractor.aliked_n32:
                return {"url": "./weights/aliked-n32.pth"}
            case Extractor.aliked_t16:
                return {
                        "input_dim": 64,
                        "url": "./weights/aliked-t16.pth"
                    }
            case Extractor.disk:
                return {
                    "input_dim": 128,
                    "url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth",
                }
            case Extractor.aliked:
                return {
                    "input_dim": 128,
                    "url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth",
                }