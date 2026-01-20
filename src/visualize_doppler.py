import numpy as np
from utils.visualization import show_radar_image

doppler_cpi = np.load("../data/processed/cpi_doppler/cpi_00010.npy")

show_radar_image(
    doppler_cpi,
    title="Range-Doppler CPI (Magnitude)"
)
