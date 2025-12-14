import cv2
import numpy as np
import pytest

from pyCell.domain.read.cell_img import CellImage, read_img
from pyCell.domain.read.image_path import ImagePath


def test_read_img_success():
    """
    Tests that read_img correctly reads an image file and returns a
    grayscale CellImage object.
    """
    # Use the existing sample image
    path_str = "sample.jpeg"
    cell_image = read_img(path_str)

    # Assertions
    assert isinstance(cell_image, CellImage)
    assert cell_image.img_path.value == path_str
    assert isinstance(cell_image.img, np.ndarray)
    assert cell_image.img.ndim == 2  # Should be grayscale
    assert cell_image.img.shape[0] > 0
    assert cell_image.img.shape[1] > 0


def test_read_img_not_found():
    """
    Tests that read_img raises FileNotFoundError for a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        read_img("non_existent_file.jpg")


def test_read_img_corrupted_file(tmp_path):
    """
    Tests that read_img raises ValueError if cv2 cannot read the image.
    """
    # Create an empty file that is not a valid image
    p = tmp_path / "corrupted.jpg"
    p.touch()

    with pytest.raises(ValueError, match="画像の読み込みに失敗しました"):
        read_img(str(p))


def test_cell_image_preprocessing_methods():
    """
    Tests that preprocessing methods like clahe and gaussian_blur
    return a new CellImage object with modified data.
    """
    # Create a dummy CellImage correctly
    initial_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    dummy_path = ImagePath("dummy.jpg")
    cell_image = CellImage(img_path=dummy_path, img=initial_data)

    # Test CLAHE
    clahe_image = cell_image.clahe(clipLimit=2.0, tileGridSize=(8, 8))
    assert isinstance(clahe_image, CellImage)
    assert clahe_image is not cell_image
    assert clahe_image.img_path is cell_image.img_path
    assert not np.array_equal(clahe_image.img, initial_data)

    # Test Gaussian Blur
    blurred_image = cell_image.gaussian_blur(ksize=(5, 5))
    assert isinstance(blurred_image, CellImage)
    assert blurred_image is not cell_image
    assert blurred_image.img_path is cell_image.img_path
    assert not np.array_equal(blurred_image.img, initial_data)
