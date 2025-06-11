import torch
import sys
sys.path.append('/workspace/PolarRTDETRv2')
from src.data.dataset.coco_dataset import *

# Try to load one sample and see landmarks structure
print("Testing landmark tensor structure...")
