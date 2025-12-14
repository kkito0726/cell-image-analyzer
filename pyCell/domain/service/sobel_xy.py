from dataclasses import dataclass

import cv2

from pyCell.domain.read.cell_img import CellImage


@dataclass(frozen=True)
class SobelXY:
    ix: cv2.typing.MatLike
    iy: cv2.typing.MatLike


def sobel_xy_factory(cell_img: CellImage, ddepth=cv2.CV_64F, ksize=3) -> SobelXY:
    return SobelXY(
        cell_img.sobel(1, 0, ddepth, ksize=ksize).img,
        cell_img.sobel(0, 1, ddepth, ksize=ksize).img,
    )
