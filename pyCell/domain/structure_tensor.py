from dataclasses import dataclass
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyCell.domain.filter.sobel import Sobel

@dataclass(frozen=True)
class StructureTensor:
    jxx: cv2.typing.MatLike
    jyy: cv2.typing.MatLike
    jxy: cv2.typing.MatLike

    @property
    def theta(self):
        return 0.5 * np.arctan2(2*self.jxy, self.jxx - self.jyy)

    @property
    def theta_corr(self):
        return (self.theta - np.pi/2) % np.pi   # [0, π)

    @property
    def orientation_order_parameter(self) -> np.ndarray:
        """
        秩序パラメータ（Orientation Order Parameter）\n
        S=⟨cos(2(θ−θˉ))⟩ \n
        - S ≈ 1 : 完全配向 \n
        - S ≈ 0 : ランダム \n
        """
        mean_theta = 0.5 * np.arctan2(
            np.mean(np.sin(2 * self.theta_corr)),
            np.mean(np.cos(2 * self.theta_corr))
        )
        return np.mean(np.cos(2*(self.theta_corr - mean_theta)))


    def rose_hist(self, bins=36, theta_min=0, theta_max=180):
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, polar=True)
        ax.hist(self.theta_corr.flatten(), bins=bins)
        ax.set_thetamin(theta_min)
        ax.set_thetamax(theta_max)

        ax.set_yticklabels([])
        plt.show()
        return ax

def calc_structure_tensor(sobel: Sobel) -> StructureTensor:
    return StructureTensor(
        cv2.GaussianBlur(sobel.ix * sobel.ix, (15, 15), 0),
        cv2.GaussianBlur(sobel.iy * sobel.iy, (15, 15), 0),
        cv2.GaussianBlur(sobel.ix * sobel.iy, (15, 15), 0)
    )
