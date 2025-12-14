import numpy as np

from pyCell.domain.service.sobel_xy import sobel_xy_factory
from pyCell.domain.service.structure_tensor import structure_tensor_factory


def test_structure_tensor_vertical_edge(vertical_edge_image):
    """
    Tests the structure tensor for a vertical edge.
    A vertical edge has an orientation of 90 degrees (pi / 2).
    The gradient is horizontal, so the gradient angle theta should be 0.
    """
    # Get Sobel gradients
    sobel_op = sobel_xy_factory(vertical_edge_image)

    # Calculate structure tensor
    st = structure_tensor_factory(sobel_op)

    # The interesting region is along the vertical edge
    edge_region_theta = st.theta[:, 48:52]

    # The angle of the gradient should be 0
    # We check that the values are close to 0
    assert np.all(np.abs(edge_region_theta) < 1e-6)


def test_structure_tensor_horizontal_edge(horizontal_edge_image):
    """
    Tests the structure tensor for a horizontal edge.
    A horizontal edge has an orientation of 0 degrees (0).
    The gradient is vertical, so the gradient angle theta should be +/- 90 degrees (pi / 2).
    """
    # Get Sobel gradients
    sobel_op = sobel_xy_factory(horizontal_edge_image)

    # Calculate structure tensor
    st = structure_tensor_factory(sobel_op)

    # The interesting region is along the horizontal edge
    edge_region_theta = st.theta[48:52, :]

    # The angle of the gradient should be pi/2 or -pi/2
    # We check that the absolute value is close to pi/2
    assert np.all(np.abs(np.abs(edge_region_theta) - np.pi / 2) < 1e-6)
