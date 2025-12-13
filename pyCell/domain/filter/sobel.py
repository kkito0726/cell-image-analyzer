from dataclasses import dataclass

import cv2

from pyCell.domain.image.cell_img import CellImage


@dataclass(frozen=True)
class Sobel:
    ix: cv2.typing.MatLike
    iy: cv2.typing.MatLike


def sobel_filter(img: CellImage) -> Sobel:
    return Sobel(
        cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3),
        cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3),
    )
