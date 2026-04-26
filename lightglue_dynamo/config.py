import sys
from enum import auto
from typing import Any

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name


class InferenceDevice(StrEnum):
    cpu = auto()
    cuda = auto()
    tensorrt = auto()
    openvino = auto()


class Extractor(StrEnum):
    superpoint = auto()
    superpoint_open = auto()
    disk = auto()
    aliked = auto()
    aliked_n16 = auto()
    aliked_n16rot = auto()
    aliked_n32 = auto()
    aliked_t16 = auto()
    xfeat = auto()
    raco = auto()

    @property
    def input_dim_divisor(self) -> int:
        match self:
            case Extractor.superpoint | Extractor.superpoint_open:
                return 8
            case Extractor.disk:
                return 16
            case (
                Extractor.aliked
                | Extractor.aliked_n16
                | Extractor.aliked_n16rot
                | Extractor.aliked_n32
                | Extractor.aliked_t16
            ):
                return 32
            case Extractor.xfeat:
                return 32
            case Extractor.raco:
                return 32

    @property
    def input_channels(self) -> int:
        match self:
            case Extractor.superpoint | Extractor.superpoint_open:
                return 1
            case Extractor.disk | Extractor.aliked | Extractor.aliked_n16 | Extractor.aliked_n16rot | Extractor.aliked_n32 | Extractor.aliked_t16 | Extractor.xfeat | Extractor.raco:
                return 3

    @property
    def lightglue_config(self) -> dict[str, Any]:
        match self:
            case Extractor.superpoint | Extractor.superpoint_open:
                return {"url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"}
            case Extractor.disk:
                return {
                    "input_dim": 128,
                    "url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth",
                }
            case Extractor.aliked | Extractor.aliked_n16 | Extractor.aliked_n16rot | Extractor.aliked_n32 | Extractor.aliked_t16:
                return {
                    "input_dim": 128,
                    "url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth",
                }
            case Extractor.xfeat:
                return {"input_dim": 64}
            case Extractor.raco:
                return {"input_dim": 128}
