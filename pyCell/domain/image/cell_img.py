from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt

from pyCell.domain.image.image_path import ImagePath


@dataclass(frozen=True)
class CellImage:
    img_path: ImagePath
    img: cv2.typing.MatLike

    def show(self):
        plt.figure()
        plt.imshow(self.img)
        plt.axis("off")
        plt.show()

    def clahe(self, clipLimit=2.0, tileGridSize=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        img_eq = clahe.apply(self.img)

        return CellImage(self.img_path, img_eq)

    def gaussian_blur(self, ksize=(5, 5), sigmaX=0):
        img_blur = cv2.GaussianBlur(self.img, ksize=ksize, sigmaX=sigmaX)

        return CellImage(self.img_path, img_blur)


def read_img(img_path: str) -> CellImage:
    img_path = ImagePath(img_path)
    img = cv2.imread(img_path.value, cv2.IMREAD_GRAYSCALE)

    return CellImage(img_path, img)
