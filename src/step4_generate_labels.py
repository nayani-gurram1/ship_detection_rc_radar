import os
import json
import numpy as np
from tqdm import tqdm
from utils.cfar import ca_cfar_2d
from utils.bbox import detections_to_bboxes

INPUT_DIR = "../data/processed/cpi_doppler"
LABEL_DIR = "../data/labels"

os.makedirs(LABEL_DIR, exist_ok=True)

def main():
    cpi_files = sorted(os.listdir(INPUT_DIR))
    print("Total Doppler CPIs:", len(cpi_files))

    for file in tqdm(cpi_files):
        cpi = np.load(os.path.join(INPUT_DIR, file))

        detections = ca_cfar_2d(
            cpi,
            guard_cells=(2, 2),
            training_cells=(8, 8),
            threshold_scale=3.0
        )

        bboxes = detections_to_bboxes(detections)

        label_data = {
                        "image": file,
                        "bboxes": [[int(v) for v in box] for box in bboxes],
                        "class": "ship"
                    }


        label_file = file.replace(".npy", ".json")
        with open(os.path.join(LABEL_DIR, label_file), "w") as f:
            json.dump(label_data, f)

    print("Labels saved to:", LABEL_DIR)

if __name__ == "__main__":
    main()
