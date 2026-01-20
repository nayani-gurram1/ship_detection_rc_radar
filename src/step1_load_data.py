from utils.radar_io import load_radar_data
from utils.visualization import show_radar_image

RADAR_FILE = "../data/raw/rc_radar_sample.npy"

def main():
    radar_data = load_radar_data(RADAR_FILE)

    print("Radar data loaded successfully")
    print("Shape (Azimuth, Range):", radar_data.shape)

    show_radar_image(
        radar_data[:2000, :2000],
        title="Range-Compressed Radar Data (Sample)"
    )

if __name__ == "__main__":
    main()
