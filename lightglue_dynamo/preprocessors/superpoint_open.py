import numpy as np

from .base import PreprocessorBase

class SuperPointOpenPreprocessor(PreprocessorBase):
    @staticmethod
    def preprocess(image_batch: np.ndarray) -> np.ndarray:
        """
        Preprocess a batch of images from cv2.imread format (N, H, W, 3), BGR.
        - Converts BGR to RGB, then to grayscale.
        - Normalizes to [0.0, 1.0].
        - Pads height and width to be divisible by 8.
        - Transposes to (N, 1, H_new, W_new).
        """
        if image_batch.ndim == 3:
            image_batch = image_batch[np.newaxis, ...]

        if image_batch.shape[-1] == 3:
            rgb_batch = image_batch[..., ::-1]
            gray_batch = np.dot(rgb_batch, [0.299, 0.587, 0.114])
        else: 
            gray_batch = image_batch.squeeze(axis=-1)
            
        gray_batch = gray_batch.astype(np.float32) / 255.0

        N, H, W = gray_batch.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        
        padded_batch = np.pad(
            gray_batch, 
            pad_width=((0, 0), (0, pad_h), (0, pad_w)),
            mode='constant', 
            constant_values=0
        )

        return padded_batch[:, np.newaxis, :, :]