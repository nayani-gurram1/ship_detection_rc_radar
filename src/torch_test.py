# torch_test.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU before importing torch

import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
