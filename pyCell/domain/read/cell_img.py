import os
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt

from pyCell.domain.read.image_path import ImagePath


@dataclass(frozen=True)
class CellImage:
    img_path: ImagePath
    img: cv2.typing.MatLike

    def show(self):
        plt.figure()
        plt.imshow(self.img, cmap="gray")
        plt.axis("off")
        plt.show()

    def save(self, file_path: str):
        """
        Saves the current image data to a file.

        Args:
            file_path: The path where the image will be saved.
                       The file extension must be a valid image format
                       (e.g., .png, .jpg, .bmp).
        """
        # Prevent overwriting the source file
        if os.path.abspath(file_path) == os.path.abspath(self.img_path.value):
            raise ValueError(
                "読み込み中の画像ファイルとファイルパスが一致しています"
            )
        
        # Validate the output path using ImagePath
        output_path = ImagePath(file_path)
        
        # Save the image using OpenCV
        cv2.imwrite(output_path.value, self.img)

    def clahe(self, clipLimit=2.0, tileGridSize=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        img_eq = clahe.apply(self.img)

        return CellImage(self.img_path, img_eq)

    def gaussian_blur(self, ksize=(5, 5), sigmaX=0):
        img_blur = cv2.GaussianBlur(self.img, ksize=ksize, sigmaX=sigmaX)

        return CellImage(self.img_path, img_blur)

    def sobel(self, dx: int, dy: int, ddepth=cv2.CV_64F, ksize=3):
        return CellImage(
            self.img_path, cv2.Sobel(self.img, ddepth, dx=dx, dy=dy, ksize=ksize)
        )


def read_img(img_path: str) -> CellImage:
    image_path = ImagePath(img_path)
    if not os.path.exists(image_path.value):
        raise FileNotFoundError(f"ファイルが見つかりません: {image_path.value}")

    img = cv2.imread(image_path.value, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"画像の読み込みに失敗しました: {image_path.value}")

    return CellImage(image_path, img)
