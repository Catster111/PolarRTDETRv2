"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Custom collate function for face detection with landmarks
"""

import torch
import torch.nn.functional as F
import random
from typing import List, Dict, Any

from .dataloader import BaseCollateFunction
from ..core import register

__all__ = ['FaceLandmarkCollateFuncion', 'face_landmark_collate_fn']


@register()
class FaceLandmarkCollateFuncion(BaseCollateFunction):
    """Custom collate function for face detection with landmarks and polar rotations"""
    
    def __init__(
        self, 
        scales=None, 
        stop_epoch=None,
        heatmap_interpolation='bilinear'
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.heatmap_interpolation = heatmap_interpolation

    def __call__(self, items: List) -> tuple:
        """Collate batch of face detection samples"""
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        # Apply multi-scale training if enabled
        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            original_size = images.shape[-2:]
            
            # Resize images
            images = F.interpolate(images, size=sz, mode='bilinear', align_corners=False)
            
            # Update targets for new image size
            scale_factor_h = sz / original_size[0] if isinstance(sz, int) else sz[0] / original_size[0]
            scale_factor_w = sz / original_size[1] if isinstance(sz, int) else sz[1] / original_size[1]
            
            for target in targets:
                # Update image size in target
                if isinstance(sz, int):
                    target['size'] = torch.tensor([sz, sz])
                else:
                    target['size'] = torch.tensor([sz[0], sz[1]])
                
                # Scale landmark heatmaps if present
                if 'landmark_heatmaps' in target and target['landmark_heatmaps'].numel() > 0:
                    heatmaps = target['landmark_heatmaps']
                    
                    if heatmaps.dim() == 4 and heatmaps.size(0) > 0:
                        new_heatmap_size = int(64 * min(scale_factor_h, scale_factor_w))
                        new_heatmap_size = max(16, min(128, new_heatmap_size))
                        
                        resized_heatmaps = F.interpolate(
                            heatmaps.view(-1, heatmaps.size(-2), heatmaps.size(-1)).unsqueeze(1),
                            size=new_heatmap_size,
                            mode=self.heatmap_interpolation,
                            align_corners=False
                        ).squeeze(1).view(heatmaps.size(0), 5, new_heatmap_size, new_heatmap_size)
                        
                        target['landmark_heatmaps'] = resized_heatmaps
                
        return images, targets


@register()
def face_landmark_collate_fn(items: List) -> tuple:
    """Simple collate function for face landmark detection"""
    images = torch.cat([x[0][None] for x in items], dim=0)
    targets = [x[1] for x in items]
    return images, targets
