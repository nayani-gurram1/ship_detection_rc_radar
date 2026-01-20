import numpy as np

def load_radar_data(file_path):
    """
    Load range-compressed radar data.
    Expected shape: (azimuth_samples, range_bins)
    """
    data = np.load(file_path)

    if data.ndim != 2:
        raise ValueError("Radar data must be 2D")

    return data
