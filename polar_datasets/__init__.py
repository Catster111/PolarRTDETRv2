"""
Datasets module for Polar-RTDETRv2.

This module provides dataset implementations for training and evaluating 
the Polar-RTDETRv2 model, with a focus on face detection with landmarks.
"""

from .widerface import (
    WiderFaceDataset,
    WiderFaceLandmarksDataset,
    collate_fn,
    build_widerface
)
from .transforms import (
    RandomHorizontalFlip,
    RandomResize,
    RandomSelect,
    RandomSizeCrop,
    Normalize,
    Compose,
    ToTensor,
    LandmarkAugmentation
)
from .coco import build as build_coco

from .data_prefetcher import data_prefetcher
from .samplers import DistributedSampler, NodeDistributedSampler

__all__ = [
    # WiderFace dataset
    'WiderFaceDataset',
    'WiderFaceLandmarksDataset',
    'collate_fn',
    'build_widerface',
    
    # Transforms
    'RandomHorizontalFlip',
    'RandomResize',
    'RandomSelect',
    'RandomSizeCrop',
    'Normalize',
    'Compose',
    'ToTensor',
    'LandmarkAugmentation',
    
    # COCO dataset (for compatibility)
    'build_coco',
    
    # Utilities
    'data_prefetcher',
    'DistributedSampler',
    'NodeDistributedSampler',
]
