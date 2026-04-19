import numpy as np

from .base import PreprocessorBase


class ALIKEDPreprocessor(PreprocessorBase):
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Preprocess image from cv2.imread, (..., H, W, 3), BGR."""
        # convert BGR to RGB
        image = image[..., ::-1] 
        
        image = image / 255
        axes = [*list(range(image.ndim - 3)), -1, -3, -2]
        image = image.transpose(*axes)  # (..., 3, H, W)
        
        return image