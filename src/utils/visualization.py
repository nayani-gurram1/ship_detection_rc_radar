import numpy as np
import matplotlib.pyplot as plt

def show_radar_image(data, title="Radar Image"):
    magnitude = np.abs(data)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        20 * np.log10(magnitude + 1e-6),
        aspect='auto',
        cmap='gray'
    )
    plt.colorbar(label="Intensity (dB)")
    plt.xlabel("Range Bins")
    plt.ylabel("Azimuth Samples")
    plt.title(title)
    plt.show()
