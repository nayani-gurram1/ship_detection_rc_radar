import os
import numpy as np
import matplotlib.pyplot as plt
import json

# Paths
CPI_DIR = "../data/processed/cpi_time"       # or cpi_doppler
LABEL_DIR = "../data/labels"

def show_cpi_with_bboxes(cpi_file, label_file):
    # Load CPI
    cpi = np.load(cpi_file)
    # Load labels
    with open(label_file, 'r') as f:
        labels = json.load(f)

    plt.figure(figsize=(12,6))
    plt.imshow(np.abs(cpi), cmap='gray')
    plt.title(f"CPI: {os.path.basename(cpi_file)}")
    
    # Draw bounding boxes
    for bbox in labels['bboxes']:
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min),
                             x_max-x_min,
                             y_max-y_min,
                             edgecolor='red',
                             facecolor='none',
                             linewidth=2)
        plt.gca().add_patch(rect)
    
    plt.show()

def main():
    cpi_files = sorted(os.listdir(CPI_DIR))
    label_files = sorted(os.listdir(LABEL_DIR))
    
    # Show first 5 CPIs for verification
    for cpi_file, label_file in zip(cpi_files[:5], label_files[:5]):
        show_cpi_with_bboxes(os.path.join(CPI_DIR, cpi_file),
                             os.path.join(LABEL_DIR, label_file))

if __name__ == "__main__":
    main()
