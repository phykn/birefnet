import base64
import binascii
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 36_000_000
MAX_BASE64_LENGTH = 4 * ((MAX_BYTES + 2) // 3)


class ImageLimitError(ValueError):
    pass


def decode(text: str) -> np.ndarray:
    if len(text) > MAX_BASE64_LENGTH:
        raise ImageLimitError("image file is too large")
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64") from exc

    if len(raw) > MAX_BYTES:
        raise ImageLimitError("image file is too large")
    try:
        with Image.open(BytesIO(raw)) as source:
            width, height = source.size
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError) as exc:
        raise ValueError("invalid image") from exc
    if width * height > MAX_PIXELS:
        raise ImageLimitError("image dimensions are too large")

    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None:
        raise ValueError("invalid image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode(mask: np.ndarray, indexed: bool = False) -> str:
    if indexed:
        image = Image.fromarray(mask)
        palette = np.repeat(np.arange(256, dtype=np.uint8), 3).tolist()
        palette[:12] = [24, 32, 48, 174, 187, 199, 239, 89, 79, 64, 184, 220]
        image.putpalette(palette)
        output = BytesIO()
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")
    ok, raw = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("Failed to encode prediction")
    return base64.b64encode(raw).decode("ascii")
