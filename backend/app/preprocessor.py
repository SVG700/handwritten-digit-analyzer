"""
Preprocessing module for canvas drawings and uploaded images.
Converts arbitrary user images into an 8x8 grayscale matrix compatible with
the scikit-learn digits dataset (intensities in [0, 16], stroke bright on dark background).
"""

import base64
import io
import re
from typing import Tuple
import numpy as np
from PIL import Image, ImageOps


def base64_to_image(data_url: str) -> Image.Image:
    """
    Decodes a base64 data URL string into a PIL Image.
    Accepts data like 'data:image/png;base64,...' or raw base64 strings.
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_data = base64.b64decode(data_url)
    return Image.open(io.BytesIO(image_data))


def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """Loads a PIL Image from raw bytes (uploaded file)."""
    return Image.open(io.BytesIO(image_bytes))


def preprocess_image_to_8x8(image: Image.Image) -> Tuple[np.ndarray, bool]:
    """
    Preprocesses an input PIL image into an 8x8 grayscale NumPy matrix:
    1. Converts image to RGBA / Grayscale.
    2. Handles alpha channels (transparent background converted to appropriate tone).
    3. Detects background vs foreground polarity:
       Ensures the digit stroke is bright on a dark background (0 to 16 scale).
    4. Bounding-box detection: crops tightly around the digit.
    5. Aspect-ratio preserving padding to make the crop square.
    6. Downsamples to 8x8 with anti-aliasing.
    7. Scales pixel intensities from [0, 255] to [0.0, 16.0].
    
    Returns:
        Tuple of (8x8 numpy array, is_blank_boolean).
    """
    # Handle transparency
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        alpha = image.convert("RGBA").split()[-1]
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=alpha)
        image = bg

    # Convert to grayscale
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # Check polarity: background vs foreground
    # If four corners or edges are predominantly light (> 127), invert image
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    if np.mean(corners) > 127 or np.mean(arr) > 127:
        arr = 255.0 - arr

    # Threshold for noise cleanup
    threshold = 20.0
    arr[arr < threshold] = 0.0

    # Check if image is blank / all zeros
    if np.max(arr) < 1.0:
        # Blank canvas
        return np.zeros((8, 8), dtype=np.float64), True

    # Find bounding box of foreground pixels
    non_zero_coords = np.argwhere(arr > threshold)
    if len(non_zero_coords) == 0:
        return np.zeros((8, 8), dtype=np.float64), True

    ymin, xmin = non_zero_coords.min(axis=0)
    ymax, xmax = non_zero_coords.max(axis=0)

    # Add 10% padding around bounding box
    h = ymax - ymin + 1
    w = xmax - xmin + 1
    pad = int(max(h, w) * 0.15)

    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    ymax = min(arr.shape[0] - 1, ymax + pad)
    xmax = min(arr.shape[1] - 1, xmax + pad)

    cropped = arr[ymin : ymax + 1, xmin : xmax + 1]

    # Make square while preserving center of mass
    ch, cw = cropped.shape
    max_dim = max(ch, cw)
    square_img = np.zeros((max_dim, max_dim), dtype=np.float32)

    y_offset = (max_dim - ch) // 2
    x_offset = (max_dim - cw) // 2
    square_img[y_offset : y_offset + ch, x_offset : x_offset + cw] = cropped

    # Resize to 8x8 using PIL anti-aliased resampling
    pil_square = Image.fromarray(square_img.astype(np.uint8))
    pil_8x8 = pil_square.resize((8, 8), resample=Image.Resampling.LANCZOS)
    arr_8x8 = np.array(pil_8x8, dtype=np.float64)

    # Scale from [0, 255] to [0, 16] to match scikit-learn digits scale
    if np.max(arr_8x8) > 0:
        # Normalize relative contrast to match typical digit range [0, 16]
        arr_8x8 = (arr_8x8 / np.max(arr_8x8)) * 16.0
    else:
        arr_8x8 = np.zeros((8, 8), dtype=np.float64)

    return arr_8x8, False
