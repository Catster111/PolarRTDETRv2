"""
Box operations for Polar-RTDETRv2.

This module provides utility functions for box operations, including:
- Coordinate conversions (xyxy <-> cxcywh)
- IoU calculations
- Generalized IoU calculations
- Polar coordinate conversions
"""

import torch
import math
from typing import Tuple


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box format from [cx, cy, w, h] to [x1, y1, x2, y2].
    
    Args:
        x: Bounding box tensor in [cx, cy, w, h] format
        
    Returns:
        Bounding box tensor in [x1, y1, x2, y2] format
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding box format from [x1, y1, x2, y2] to [cx, cy, w, h].
    
    Args:
        x: Bounding box tensor in [x1, y1, x2, y2] format
        
    Returns:
        Bounding box tensor in [cx, cy, w, h] format
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    """
    Compute area of boxes.
    
    Args:
        boxes: Boxes in [x1, y1, x2, y2] format
        
    Returns:
        area: Area of boxes
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: First set of boxes in [x1, y1, x2, y2] format
        boxes2: Second set of boxes in [x1, y1, x2, y2] format
        
    Returns:
        iou: IoU between boxes
        union: Union area between boxes
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Compute generalized IoU between two sets of boxes.
    
    Args:
        boxes1: First set of boxes in [x1, y1, x2, y2] format
        boxes2: Second set of boxes in [x1, y1, x2, y2] format
        
    Returns:
        giou: Generalized IoU between boxes
    """
    # Calculate IoU
    iou, union = box_iou(boxes1, boxes2)
    
    # Calculate enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]
    
    # Calculate GIoU
    giou = iou - (area - union) / area.clamp(min=1e-6)
    
    return giou


def box_diou(boxes1, boxes2):
    """
    Compute Distance-IoU between two sets of boxes.
    
    Args:
        boxes1: First set of boxes in [x1, y1, x2, y2] format
        boxes2: Second set of boxes in [x1, y1, x2, y2] format
        
    Returns:
        diou: Distance-IoU between boxes
    """
    # Calculate IoU
    iou, _ = box_iou(boxes1, boxes2)
    
    # Calculate centers
    centers1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2
    centers2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2
    
    # Calculate squared distance between centers
    dist = torch.sum((centers1[:, None, :] - centers2[None, :, :]) ** 2, dim=-1)
    
    # Calculate diagonal length of smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    c2 = torch.sum(wh ** 2, dim=-1)
    
    # Calculate DIoU
    diou = iou - dist / c2.clamp(min=1e-6)
    
    return diou


def box_ciou(boxes1, boxes2):
    """
    Compute Complete-IoU between two sets of boxes.
    
    Args:
        boxes1: First set of boxes in [x1, y1, x2, y2] format
        boxes2: Second set of boxes in [x1, y1, x2, y2] format
        
    Returns:
        ciou: Complete-IoU between boxes
    """
    # Calculate DIoU
    diou = box_diou(boxes1, boxes2)
    
    # Calculate aspect ratio term
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]
    
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w1 / h1.clamp(min=1e-6))[:, None] - torch.atan(w2 / h2.clamp(min=1e-6)), 2)
    alpha = v / (1 - diou + v).clamp(min=1e-6)
    
    # Calculate CIoU
    ciou = diou - alpha * v
    
    return ciou


def polar_to_cartesian_box(boxes, image_size=None):
    """
    Convert boxes from polar [r1, theta1, r2, theta2] to Cartesian [x1, y1, x2, y2] coordinates.
    
    Args:
        boxes: Boxes in polar coordinates [N, 4]
        image_size: Image size (height, width) for normalization
        
    Returns:
        cart_boxes: Boxes in Cartesian coordinates [N, 4]
    """
    # Get center of image
    if image_size is not None:
        height, width = image_size
        center_x, center_y = width / 2, height / 2
        diag = torch.sqrt(torch.tensor(width**2 + height**2, device=boxes.device))
    else:
        # Assume normalized coordinates [0, 1]
        center_x, center_y = 0.5, 0.5
        diag = torch.sqrt(torch.tensor(2.0, device=boxes.device))
    
    # Extract coordinates
    r1, theta1, r2, theta2 = boxes.unbind(-1)
    
    # Denormalize radius if needed
    if image_size is not None:
        r1 = r1 * diag
        r2 = r2 * diag
    
    # Calculate Cartesian coordinates for top-left corner
    x1 = center_x + r1 * torch.cos(theta1)
    y1 = center_y + r1 * torch.sin(theta1)
    
    # Calculate Cartesian coordinates for bottom-right corner
    x2 = center_x + r2 * torch.cos(theta2)
    y2 = center_y + r2 * torch.sin(theta2)
    
    # Ensure proper ordering (x1 <= x2, y1 <= y2)
    x_min = torch.min(x1, x2)
    y_min = torch.min(y1, y2)
    x_max = torch.max(x1, x2)
    y_max = torch.max(y1, y2)
    
    # Normalize coordinates if needed
    if image_size is not None:
        x_min = x_min / width
        y_min = y_min / height
        x_max = x_max / width
        y_max = y_max / height
    
    # Stack to get Cartesian boxes
    cart_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    
    return cart_boxes


def cartesian_to_polar_box(boxes, image_size=None):
    """
    Convert boxes from Cartesian [x1, y1, x2, y2] to polar [r1, theta1, r2, theta2] coordinates.
    
    Args:
        boxes: Boxes in Cartesian coordinates [N, 4]
        image_size: Image size (height, width) for normalization
        
    Returns:
        polar_boxes: Boxes in polar coordinates [N, 4]
    """
    # Get center of image
    if image_size is not None:
        height, width = image_size
        center_x, center_y = width / 2, height / 2
        diag = torch.sqrt(torch.tensor(width**2 + height**2, device=boxes.device))
    else:
        # Assume normalized coordinates [0, 1]
        center_x, center_y = 0.5, 0.5
        diag = torch.sqrt(torch.tensor(2.0, device=boxes.device))
    
    # Extract coordinates
    x1, y1, x2, y2 = boxes.unbind(-1)
    
    # Denormalize coordinates if needed
    if image_size is not None:
        x1 = x1 * width
        y1 = y1 * height
        x2 = x2 * width
        y2 = y2 * height
    
    # Calculate polar coordinates for top-left corner
    dx1 = x1 - center_x
    dy1 = y1 - center_y
    r1 = torch.sqrt(dx1**2 + dy1**2)
    theta1 = torch.atan2(dy1, dx1)
    
    # Calculate polar coordinates for bottom-right corner
    dx2 = x2 - center_x
    dy2 = y2 - center_y
    r2 = torch.sqrt(dx2**2 + dy2**2)
    theta2 = torch.atan2(dy2, dx2)
    
    # Normalize radius if needed
    if image_size is not None:
        r1 = r1 / diag
        r2 = r2 / diag
    
    # Stack to get polar boxes
    polar_boxes = torch.stack([r1, theta1, r2, theta2], dim=-1)
    
    return polar_boxes


def polar_to_cartesian_landmarks(landmarks, image_size=None):
    """
    Convert landmarks from polar [r1, theta1, r2, theta2, ...] to Cartesian [x1, y1, x2, y2, ...] coordinates.
    
    Args:
        landmarks: Landmarks in polar coordinates [N, num_landmarks*2]
        image_size: Image size (height, width) for normalization
        
    Returns:
        cart_landmarks: Landmarks in Cartesian coordinates [N, num_landmarks*2]
    """
    # Get center of image
    if image_size is not None:
        height, width = image_size
        center_x, center_y = width / 2, height / 2
        diag = torch.sqrt(torch.tensor(width**2 + height**2, device=landmarks.device))
    else:
        # Assume normalized coordinates [0, 1]
        center_x, center_y = 0.5, 0.5
        diag = torch.sqrt(torch.tensor(2.0, device=landmarks.device))
    
    # Initialize output tensor
    cart_landmarks = torch.zeros_like(landmarks)
    
    # Convert each landmark point
    for i in range(0, landmarks.shape[1], 2):
        # Get r, theta coordinates
        r = landmarks[:, i]
        theta = landmarks[:, i+1]
        
        # Skip invisible landmarks (r=0)
        mask = (r != 0)
        
        # Denormalize radius if needed
        if image_size is not None:
            r_denorm = r * diag
        else:
            r_denorm = r
        
        # Calculate Cartesian coordinates
        x = center_x + r_denorm * torch.cos(theta)
        y = center_y + r_denorm * torch.sin(theta)
        
        # Normalize coordinates if needed
        if image_size is not None:
            x = x / width
            y = y / height
        
        # Set invisible landmarks to (0, 0) in Cartesian coordinates
        cart_landmarks[:, i] = x * mask
        cart_landmarks[:, i+1] = y * mask
    
    return cart_landmarks


def cartesian_to_polar_landmarks(landmarks, image_size=None):
    """
    Convert landmarks from Cartesian [x1, y1, x2, y2, ...] to polar [r1, theta1, r2, theta2, ...] coordinates.
    
    Args:
        landmarks: Landmarks in Cartesian coordinates [N, num_landmarks*2]
        image_size: Image size (height, width) for normalization
        
    Returns:
        polar_landmarks: Landmarks in polar coordinates [N, num_landmarks*2]
    """
    # Get center of image
    if image_size is not None:
        height, width = image_size
        center_x, center_y = width / 2, height / 2
        diag = torch.sqrt(torch.tensor(width**2 + height**2, device=landmarks.device))
    else:
        # Assume normalized coordinates [0, 1]
        center_x, center_y = 0.5, 0.5
        diag = torch.sqrt(torch.tensor(2.0, device=landmarks.device))
    
    # Initialize output tensor
    polar_landmarks = torch.zeros_like(landmarks)
    
    # Convert each landmark point
    for i in range(0, landmarks.shape[1], 2):
        # Get x, y coordinates
        x = landmarks[:, i]
        y = landmarks[:, i+1]
        
        # Skip invisible landmarks (x=0, y=0)
        mask = ~((x == 0) & (y == 0))
        
        # Denormalize coordinates if needed
        if image_size is not None:
            x_denorm = x * width
            y_denorm = y * height
        else:
            x_denorm = x
            y_denorm = y
        
        # Calculate polar coordinates
        dx = x_denorm - center_x
        dy = y_denorm - center_y
        
        # Calculate radius and angle
        r = torch.sqrt(dx**2 + dy**2)
        theta = torch.atan2(dy, dx)
        
        # Normalize radius if needed
        if image_size is not None:
            r = r / diag
        
        # Set invisible landmarks to (0, 0) in polar coordinates
        polar_landmarks[:, i] = r * mask
        polar_landmarks[:, i+1] = theta * mask
    
    return polar_landmarks
