import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from backend.codec import decode
from src.data.image import read_image


@pytest.mark.parametrize("shape", [(24, 40), (40, 24)])
@pytest.mark.parametrize("orientation", [None, 1, 2, 3, 4, 5, 6, 7, 8])
def test_decode_matches_training_pixels_with_exif_orientation(tmp_path, shape, orientation):
    height, width = shape
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[: height // 2, : width // 2] = [255, 0, 0]
    pixels[: height // 2, width // 2 :] = [0, 255, 0]
    pixels[height // 2 :, : width // 2] = [0, 0, 255]
    pixels[height // 2 :, width // 2 :] = [255, 255, 0]
    output = BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    Image.fromarray(pixels).save(output, format="JPEG", quality=95, exif=exif)
    raw = output.getvalue()
    path = tmp_path / "image.jpg"
    path.write_bytes(raw)

    expected = read_image(str(path))
    actual = decode(base64.b64encode(raw).decode("ascii"))

    assert actual.shape == expected.shape == (height, width, 3)
    np.testing.assert_allclose(actual, expected, atol=2)
