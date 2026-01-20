import numpy as np
from scipy.ndimage import uniform_filter

def ca_cfar_2d(
    data,
    guard_cells=(2, 2),
    training_cells=(8, 8),
    threshold_scale=3.0
):
    """
    2D CA-CFAR detector
    """
    mag = np.abs(data)

    # Total window size
    win_row = 2 * (guard_cells[0] + training_cells[0]) + 1
    win_col = 2 * (guard_cells[1] + training_cells[1]) + 1

    # Guard window size
    guard_row = 2 * guard_cells[0] + 1
    guard_col = 2 * guard_cells[1] + 1

    # Mean over training + guard
    mean_all = uniform_filter(mag, size=(win_row, win_col))

    # Mean over guard cells
    mean_guard = uniform_filter(mag, size=(guard_row, guard_col))

    # Training cells power estimate
    training_mean = (
        mean_all * win_row * win_col -
        mean_guard * guard_row * guard_col
    ) / (
        win_row * win_col - guard_row * guard_col
    )

    threshold = training_mean * threshold_scale

    detections = mag > threshold
    return detections
