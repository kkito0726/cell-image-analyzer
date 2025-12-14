from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from pyCell.domain.service.sobel_xy import SobelXY


@dataclass(frozen=True)
class StructureTensor:
    jxx: cv2.typing.MatLike
    jyy: cv2.typing.MatLike
    jxy: cv2.typing.MatLike

    @property
    def theta(self):
        """
        角度算出（rad）
        """
        return 0.5 * np.arctan2(2 * self.jxy, self.jxx - self.jyy)

    @property
    def theta_corr(self):
        """
        横方向を基準で角度算出
        """
        return (self.theta - np.pi / 2) % np.pi  # [0, π)

    @property
    def orientation_order_parameter(
        self,
    ) -> float:  # Return type changed to float as it's a single scalar
        """
        配向秩序パラメータ（Orientation Order Parameter）\n
        S = √(<cos(2θ)>^2 + <sin(2θ)>^2)\n
        - S ≈ 1 : 完全配向\n
        - S ≈ 0 : ランダム\n
        """
        cos2theta = np.cos(2 * self.theta_corr)
        sin2theta = np.sin(2 * self.theta_corr)

        mean_cos2theta = np.mean(cos2theta)
        mean_sin2theta = np.mean(sin2theta)

        return np.sqrt(mean_cos2theta**2 + mean_sin2theta**2)

    def rose_hist(
        self, bins=36, theta_min=0, theta_max=180, isShow=True, dpi=300
    ) -> Axes | None:
        """
        :param bins: ビンの数
        :param theta_min: 開始角度
        :param theta_max: 終了角度
        :param isShow: グラフ表示するかどうか
        :param dpi: グラフの解像度
        :return:
        """
        plt.figure(dpi=dpi)
        ax = plt.subplot(111, polar=True)
        ax.hist(self.theta_corr.flatten(), bins=bins)
        ax.set_thetamin(theta_min)
        ax.set_thetamax(theta_max)

        ax.set_yticklabels([])
        if isShow:
            plt.tight_layout()
            plt.show()
            return
        return ax


def structure_tensor_factory(sobelXY: SobelXY, ksize=15, sigmaX=0) -> StructureTensor:
    return StructureTensor(
        cv2.GaussianBlur(sobelXY.ix * sobelXY.ix, (ksize, ksize), sigmaX),
        cv2.GaussianBlur(sobelXY.iy * sobelXY.iy, (ksize, ksize), sigmaX),
        cv2.GaussianBlur(sobelXY.ix * sobelXY.iy, (ksize, ksize), sigmaX),
    )
