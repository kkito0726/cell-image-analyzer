import os
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


def test_cell_image_save(tmp_path):
    """
    Tests that the save method correctly writes the image data to a file.
    """
    # 1. Setup
    # Create a CellImage with known data
    initial_data = np.array([[0, 128], [255, 0]], dtype=np.uint8)
    dummy_path = ImagePath("dummy.png")
    cell_image = CellImage(img_path=dummy_path, img=initial_data)

    # Define an output path in the temporary directory
    save_path = tmp_path / "test_output.png"

    # 2. Execute
    cell_image.save(str(save_path))

    # 3. Verify
    # Check that the file was created
    assert save_path.exists()

    # Read the file back and check if the content is identical
    saved_data = cv2.imread(str(save_path), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(initial_data, saved_data)


def test_cell_image_save_invalid_extension():
    """
    Tests that the save method raises ValueError for an invalid file extension.
    """
    # Create a dummy CellImage
    initial_data = np.zeros((10, 10), dtype=np.uint8)
    dummy_path = ImagePath("dummy.png")
    cell_image = CellImage(img_path=dummy_path, img=initial_data)

    # Assert that saving with an invalid extension raises ValueError
    with pytest.raises(ValueError, match="画像ファイルのパスを入力してください"):
        cell_image.save("test_output.txt")


def test_cell_image_save_prevents_overwrite(tmp_path):
    """
    Tests that the save method raises ValueError when trying to overwrite
    the source file.
    """
    # 1. Setup
    # Create a real temporary image file
    source_path = tmp_path / "source.png"
    initial_data = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    cv2.imwrite(str(source_path), initial_data)

    # Read it into a CellImage object
    cell_image = read_img(str(source_path))

    # 2. Execute & Verify
    # Assert that trying to save to the same path raises a ValueError
    with pytest.raises(
        ValueError, match="読み込み中の画像ファイルとファイルパスが一致しています"
    ):
        cell_image.save(str(source_path))
