from pyCell.domain.service.sobel_xy import sobel_xy_factory
import numpy as np


def test_sobel_xy_factory_vertical_edge(vertical_edge_image):
    """
    Tests the Sobel factory with a vertical edge.
    A vertical edge should produce a strong horizontal gradient (ix)
    and a weak vertical gradient (iy).
    """
    #
    # The image is black on the left (0-49) and white on the right (50-99).
    # The gradient ix should be maximal around column 49/50.
    # The gradient iy should be close to zero everywhere.
    #

    # Apply Sobel filter
    sobel_op = sobel_xy_factory(vertical_edge_image)

    # Assertions
    ix = sobel_op.ix
    iy = sobel_op.iy

    # Check the shapes
    assert ix.shape == (100, 100)
    assert iy.shape == (100, 100)

    # Check the horizontal gradient (ix)
    # The max gradient should be at the edge
    # We check the mean of the column which should be the highest
    assert np.mean(np.abs(ix[:, 49])) > 100
    # Other areas should be close to zero
    assert np.mean(np.abs(ix[:, :48])) < 1e-6
    assert np.mean(np.abs(ix[:, 51:])) < 1e-6

    # Check the vertical gradient (iy), which should be all zero
    assert np.all(iy == 0)


def test_sobel_xy_factory_horizontal_edge(horizontal_edge_image):
    """
    Tests the Sobel factory with a horizontal edge.
    A horizontal edge should produce a strong vertical gradient (iy)
    and a weak horizontal gradient (ix).
    """
    #
    # The image is black on the top (0-49) and white on the bottom (50-99).
    # The gradient iy should be maximal around row 49/50.
    # The gradient ix should be close to zero everywhere.
    #

    # Apply Sobel filter
    sobel_op = sobel_xy_factory(horizontal_edge_image)

    # Assertions
    ix = sobel_op.ix
    iy = sobel_op.iy

    # Check the shapes
    assert ix.shape == (100, 100)
    assert iy.shape == (100, 100)

    # Check the horizontal gradient (ix), which should be all zero
    assert np.all(ix == 0)

    # Check the vertical gradient (iy)
    # The max gradient should be at the edge
    assert np.mean(np.abs(iy[49, :])) > 100
    # Other areas should be close to zero
    assert np.mean(np.abs(iy[:48, :])) < 1e-6
    assert np.mean(np.abs(iy[51:, :])) < 1e-6
