import os
import numpy as np
from utils.radar_io import load_radar_data
from tqdm import tqdm

# PARAMETERS
RADAR_FILE = "../data/raw/rc_radar_sample.npy"
OUTPUT_DIR = "../data/processed/cpi_time"
CPI_LENGTH = 128
STRIDE = 128  # no overlap (can change later)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_cpis(radar_data, cpi_length, stride):
    cpis = []
    for start in range(0, radar_data.shape[0] - cpi_length, stride):
        cpi = radar_data[start:start + cpi_length, :]
        cpis.append(cpi)
    return cpis

def main():
    radar_data = load_radar_data(RADAR_FILE)
    print("Radar data shape:", radar_data.shape)

    cpis = generate_cpis(radar_data, CPI_LENGTH, STRIDE)
    print("Total CPIs generated:", len(cpis))

    for idx, cpi in enumerate(tqdm(cpis)):
        file_path = os.path.join(OUTPUT_DIR, f"cpi_{idx:05d}.npy")
        np.save(file_path, cpi)

    print("CPIs saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
