import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU only

# Now safe to import modules that use torch
from dataset import RadarShipDataset

dataset = RadarShipDataset(
    cpi_dir="../data/processed/cpi_time",
    label_dir="../data/labels"
)

print(f"Total samples: {len(dataset)}")

# Test first sample
image, target = dataset[0]
print("Image shape:", image.shape)
print("Boxes:", target['boxes'])
print("Labels:", target['labels'])
