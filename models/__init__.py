"""
Models module for Polar-RTDETRv2.

This module provides model implementations for Polar-RTDETRv2, including:
- PolarRTDETRv2: Main model for face detection with landmarks in polar coordinates
- Backbone options (ResNet, etc.)
- Transformer components
- Detection and landmark heads
"""

import torch
import torch.nn as nn
from typing import Dict, Any

# Import model implementations
from .polar_rtdetrv2 import PolarRTDETRv2


def build_model(config: Dict[str, Any], num_classes: int) -> nn.Module:
    """
    Build model based on configuration.
    
    Args:
        config: Configuration dictionary
        num_classes: Number of classes (2 for face detection: background and face)
        
    Returns:
        model: Model instance
    """
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'polar_rtdetrv2').lower()
    
    if model_name == 'polar_rtdetrv2':
        # Create PolarRTDETRv2 model
        model = PolarRTDETRv2(
            backbone=model_config.get('backbone', {}).get('name', 'resnet50'),
            num_classes=num_classes,
            num_queries=model_config.get('transformer', {}).get('num_queries', 300),
            num_landmarks=model_config.get('landmarks', {}).get('num_landmarks', 5),
            use_polar=model_config.get('polar', {}).get('enabled', True),
            use_landmarks=model_config.get('landmarks', {}).get('enabled', True),
            dilation=model_config.get('backbone', {}).get('dilation', False),
            position_embedding=model_config.get('backbone', {}).get('position_embedding', 'sine'),
            hidden_dim=model_config.get('transformer', {}).get('hidden_dim', 256),
            nheads=model_config.get('transformer', {}).get('nheads', 8),
            num_encoder_layers=model_config.get('transformer', {}).get('enc_layers', 6),
            num_decoder_layers=model_config.get('transformer', {}).get('dec_layers', 6),
            dim_feedforward=model_config.get('transformer', {}).get('dim_feedforward', 2048),
            dropout=model_config.get('transformer', {}).get('dropout', 0.1),
            pre_norm=model_config.get('transformer', {}).get('pre_norm', False),
            return_intermediate_dec=True,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Initialize model weights
    if model_config.get('init', {}).get('xavier_gain', None) is not None:
        gain = model_config['init']['xavier_gain']
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=gain)
    
    # Freeze backbone batch normalization if needed
    if model_config.get('backbone', {}).get('freeze_backbone_bn', False):
        for m in model.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                for p in m.parameters():
                    p.requires_grad = False
    
    return model
