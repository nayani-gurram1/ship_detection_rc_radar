import numpy as np

azimuth = 8000
range_bins = 2048

noise = np.random.randn(azimuth, range_bins)
ships = np.zeros_like(noise)

ships[2000:2200, 800:900] += 10
ships[5000:5300, 1200:1300] += 12

radar = noise + ships

np.save("../data/raw/rc_radar_sample.npy", radar)
print("Dummy radar data created")
