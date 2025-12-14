from pyCell.domain.read.image_path import ImagePath
import pytest


def test_image_path_success():
    """
    Tests that ImagePath can be created successfully with a valid file extension.
    The class itself does not check for file existence.
    """
    path_str = "valid/path/image.jpeg"
    image_path = ImagePath(path_str)
    assert image_path.value == path_str


def test_image_path_non_existent_path_is_valid():
    """
    Tests that ImagePath instantiation does not fail for a non-existent path,
    as long as the extension is valid.
    """
    # This should not raise FileNotFoundError because the class doesn't check it.
    image_path = ImagePath("non_existent_file.png")
    assert image_path.value == "non_existent_file.png"


def test_image_path_invalid_extension_raises_value_error():
    """
    Tests that ImagePath raises ValueError for a path with an invalid extension.
    """
    with pytest.raises(ValueError, match="画像ファイルのパスを入力してください"):
        ImagePath("path/with/wrong_extension.txt")


def test_image_path_directory_path_raises_value_error():
    """
    Tests that ImagePath raises ValueError for a path that looks like a directory
    (i.e., has no valid image extension).
    """
    with pytest.raises(ValueError, match="画像ファイルのパスを入力してください"):
        ImagePath("/path/to/a/directory")