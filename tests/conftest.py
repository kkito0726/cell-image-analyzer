import numpy as np
import pytest

from pyCell.domain.read.cell_img import CellImage
from pyCell.domain.read.image_path import ImagePath


@pytest.fixture
def vertical_edge_image() -> CellImage:
    """
    Creates a 100x100 image with a sharp vertical edge in the middle.
    Left half is black (0), right half is white (255).
    This results in a strong horizontal gradient (ix).
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255
    return CellImage(ImagePath("dummy.jpg"), img)


@pytest.fixture
def horizontal_edge_image() -> CellImage:
    """
    Creates a 100x100 image with a sharp horizontal edge in the middle.
    Top half is black (0), bottom half is white (255).
    This results in a strong vertical gradient (iy).
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50:, :] = 255
    return CellImage(ImagePath("dummy.jpg"), img)
