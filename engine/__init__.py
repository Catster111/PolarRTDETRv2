"""
Engine module for Polar-RTDETRv2.

This module provides the core training and evaluation components for Polar-RTDETRv2,
including:
- Matcher: For matching predictions to ground truth
- SetCriterion: For computing losses
- Training utilities
- Evaluation utilities
"""

from .matcher import build_matcher, HungarianMatcher
from .criterion import SetCriterion

__all__ = [
    # Matcher
    'build_matcher',
    'HungarianMatcher',
    
    # Criterion
    'SetCriterion',
]
