import os
import numpy as np
from tqdm import tqdm

# PARAMETERS
INPUT_DIR = "../data/processed/cpi_time"
OUTPUT_DIR = "../data/processed/cpi_doppler"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def doppler_fft(cpi):
    """
    Apply Doppler FFT along azimuth (axis=0)
    """
    doppler = np.fft.fftshift(
        np.fft.fft(cpi, axis=0),
        axes=0
    )
    return doppler

def main():
    cpi_files = sorted(os.listdir(INPUT_DIR))
    print("Total CPIs:", len(cpi_files))

    for file in tqdm(cpi_files):
        cpi = np.load(os.path.join(INPUT_DIR, file))
        doppler_cpi = doppler_fft(cpi)

        save_path = os.path.join(OUTPUT_DIR, file)
        np.save(save_path, doppler_cpi)

    print("Doppler CPIs saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
