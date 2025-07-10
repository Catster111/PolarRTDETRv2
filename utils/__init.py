"""
Utilities module for Polar-RTDETRv2.

This module provides various utility functions and classes for the Polar-RTDETRv2
project, including:
- Box operations (IoU calculations, conversions)
- Distributed training utilities
- Logging and metrics
- Visualization helpers
- Miscellaneous helper functions
"""

# Import utilities to make them available
from .box_ops import (
    box_iou,
    generalized_box_iou,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
)

from .misc import (
    MetricLogger,
    SmoothedValue,
    reduce_dict,
    warmup_lr_scheduler,
    setup_logger,
    save_config,
    is_main_process,
    get_rank,
    get_world_size,
    init_distributed_mode,
    setup_for_distributed,
)

from .visualizer import Visualizer

__all__ = [
    # Box operations
    'box_iou',
    'generalized_box_iou',
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
    
    # Metrics and logging
    'MetricLogger',
    'SmoothedValue',
    'reduce_dict',
    'warmup_lr_scheduler',
    'setup_logger',
    'save_config',
    
    # Distributed utilities
    'is_main_process',
    'get_rank',
    'get_world_size',
    'init_distributed_mode',
    'setup_for_distributed',
    
    # Visualization
    'Visualizer',
]
