import numpy as np
from scipy.ndimage import label

def detections_to_bboxes(detection_map, min_pixels=20):
    labeled, num_objects = label(detection_map)
    bboxes = []

    for obj_id in range(1, num_objects + 1):
        coords = np.where(labeled == obj_id)

        if len(coords[0]) < min_pixels:
            continue

        y_min = int(coords[0].min())
        y_max = int(coords[0].max())
        x_min = int(coords[1].min())
        x_max = int(coords[1].max())

        bboxes.append([x_min, y_min, x_max, y_max])

    return bboxes
