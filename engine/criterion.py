"""
Criterion module for Polar-RTDETRv2.

This module provides the loss functions for Polar-RTDETRv2, including:
- Classification loss (focal loss or cross-entropy)
- Box regression loss (L1 and GIoU)
- Landmark regression loss (L1 for landmark coordinates)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union


class SetCriterion(nn.Module):
    """
    SetCriterion for Polar-RTDETRv2.
    
    This class computes the loss for Polar-RTDETRv2. The process happens in two steps:
    1) We compute the Hungarian assignment between ground truth and predictions
    2) We supervise each pair of matched ground-truth / prediction (including class, boxes, and landmarks)
    
    The losses include:
    - Classification loss (focal loss or cross-entropy)
    - Box regression loss (L1 and GIoU)
    - Landmark regression loss (L1 for landmark coordinates)
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher,
        weight_dict: Dict[str, float],
        losses: List[str],
        num_landmarks: int = 5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_focal: bool = True,
        use_polar: bool = False
    ):
        """
        Initialize the criterion.
        
        Args:
            num_classes: Number of object categories (excluding no-object)
            matcher: Module able to compute a matching between targets and proposals
            weight_dict: Dictionary containing weights for different losses
            losses: List of losses to apply (e.g., ['labels', 'boxes', 'landmarks'])
            num_landmarks: Number of facial landmarks
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            use_focal: Whether to use focal loss for classification
            use_polar: Whether to use polar coordinate representation
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.num_landmarks = num_landmarks
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal
        self.use_polar = use_polar
        
        # For focal loss
        if self.use_focal:
            self.register_buffer('empty_weight', torch.ones(self.num_classes + 1))
        else:
            # For cross-entropy loss, we need to handle class imbalance
            # We give higher weight to the no-object class (index 0)
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[0] = 0.1  # Lower weight for background
            self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: int) -> Dict[str, torch.Tensor]:
        """
        Classification loss (focal loss or cross-entropy).
        
        Args:
            outputs: Dictionary with model outputs
            targets: List of dictionaries with targets
            indices: List of tuples (pred_idx, tgt_idx) with indices of matched predictions and targets
            num_boxes: Number of boxes (normalization factor)
            
        Returns:
            losses: Dictionary with classification losses
        """
        assert 'pred_logits' in outputs
        
        # Get classification logits
        src_logits = outputs['pred_logits']
        
        # Get matched indices
        idx = self._get_src_permutation_idx(indices)
        
        # Get target classes
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        # Compute classification loss
        if self.use_focal:
            # Focal loss
            src_logits = src_logits.flatten(0, 1)
            target_classes = target_classes.flatten(0, 1)
            
            # Prepare one-hot target
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1] + 1],
                                               dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(1, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :-1]
            
            # Compute focal loss
            pt = src_logits.sigmoid() * target_classes_onehot + (1 - src_logits.sigmoid()) * (1 - target_classes_onehot)
            loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction='none')
            loss_ce = ((1 - pt) ** self.focal_gamma) * loss_ce
            
            # Apply alpha
            alpha = self.focal_alpha * target_classes_onehot + (1 - self.focal_alpha) * (1 - target_classes_onehot)
            loss_ce = alpha * loss_ce
            
            # Normalize by number of boxes
            loss_ce = loss_ce.sum() / num_boxes
        else:
            # Cross-entropy loss
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        losses = {'loss_ce': loss_ce}
        
        return losses
    
    def loss_boxes(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: int) -> Dict[str, torch.Tensor]:
        """
        Box regression loss (L1 and GIoU).
        
        Args:
            outputs: Dictionary with model outputs
            targets: List of dictionaries with targets
            indices: List of tuples (pred_idx, tgt_idx) with indices of matched predictions and targets
            num_boxes: Number of boxes (normalization factor)
            
        Returns:
            losses: Dictionary with box regression losses
        """
        assert 'pred_boxes' in outputs
        
        # Get predicted boxes
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        # Get target boxes
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Check if we need to convert coordinates
        use_polar_targets = any(t.get("use_polar", False) for t in targets)
        
        # Convert coordinates if needed
        if self.use_polar != use_polar_targets:
            if self.use_polar:
                # Convert target boxes from Cartesian to polar
                target_boxes = self._cartesian_to_polar(target_boxes)
            else:
                # Convert target boxes from polar to Cartesian
                target_boxes = self._polar_to_cartesian(target_boxes)
        
        # Compute L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        
        # Compute GIoU loss
        if self.use_polar:
            # For polar coordinates, convert to Cartesian for IoU calculation
            src_boxes_cart = self._polar_to_cartesian(src_boxes)
            target_boxes_cart = self._polar_to_cartesian(target_boxes) if use_polar_targets else target_boxes
            loss_giou = 1 - torch.diag(self._generalized_box_iou(src_boxes_cart, target_boxes_cart))
        else:
            loss_giou = 1 - torch.diag(self._generalized_box_iou(src_boxes, target_boxes))
        
        loss_giou = loss_giou.sum() / num_boxes
        
        losses = {
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }
        
        return losses
    
    def loss_landmarks(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices: List[Tuple[torch.Tensor, torch.Tensor]], num_boxes: int) -> Dict[str, torch.Tensor]:
        """
        Landmark regression loss (L1).
        
        Args:
            outputs: Dictionary with model outputs
            targets: List of dictionaries with targets
            indices: List of tuples (pred_idx, tgt_idx) with indices of matched predictions and targets
            num_boxes: Number of boxes (normalization factor)
            
        Returns:
            losses: Dictionary with landmark regression losses
        """
        if 'pred_landmarks' not in outputs or not all('landmarks' in t for t in targets):
            return {'loss_landmarks': torch.tensor(0.0, device=outputs['pred_logits'].device)}
        
        # Get predicted landmarks
        idx = self._get_src_permutation_idx(indices)
        src_landmarks = outputs['pred_landmarks'][idx]
        
        # Get target landmarks
        target_landmarks = torch.cat([t['landmarks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Check if we need to convert coordinates
        use_polar_targets = any(t.get("use_polar", False) for t in targets)
        
        # Convert coordinates if needed
        if self.use_polar != use_polar_targets:
            if self.use_polar:
                # Convert target landmarks from Cartesian to polar
                target_landmarks = self._cartesian_to_polar_landmarks(target_landmarks)
            else:
                # Convert target landmarks from polar to Cartesian
                target_landmarks = self._polar_to_cartesian_landmarks(target_landmarks)
        
        # Create visibility mask
        if self.use_polar:
            # In polar coordinates, radius=0 means invisible
            vis_mask = torch.ones_like(target_landmarks)
            for i in range(0, target_landmarks.shape[1], 2):
                if (target_landmarks[:, i] == 0).any():  # radius = 0 means invisible
                    vis_mask[:, i] = 0
                    vis_mask[:, i+1] = 0
        else:
            # In Cartesian coordinates, x=0 and y=0 means invisible
            vis_mask = torch.ones_like(target_landmarks)
            for i in range(0, target_landmarks.shape[1], 2):
                invisible = (target_landmarks[:, i] == 0) & (target_landmarks[:, i+1] == 0)
                vis_mask[:, i][invisible] = 0
                vis_mask[:, i+1][invisible] = 0
        
        # Compute L1 loss only for visible landmarks
        loss_landmarks = F.l1_loss(src_landmarks, target_landmarks, reduction='none') * vis_mask
        
        # Normalize by number of visible landmarks
        num_visible = vis_mask.sum()
        if num_visible > 0:
            loss_landmarks = loss_landmarks.sum() / num_visible
        else:
            loss_landmarks = loss_landmarks.sum() * 0.0  # Return 0 if no visible landmarks
        
        losses = {'loss_landmarks': loss_landmarks}
        
        return losses
    
    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get source permutation indices.
        
        Args:
            indices: List of tuples (pred_idx, tgt_idx) with indices of matched predictions and targets
            
        Returns:
            batch_idx: Batch indices
            src_idx: Source indices
        """
        # Concatenate batch dimension and source indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get target permutation indices.
        
        Args:
            indices: List of tuples (pred_idx, tgt_idx) with indices of matched predictions and targets
            
        Returns:
            batch_idx: Batch indices
            tgt_idx: Target indices
        """
        # Concatenate batch dimension and target indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def _cartesian_to_polar(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from Cartesian [x1, y1, x2, y2] to polar [r1, theta1, r2, theta2] coordinates.
        
        Args:
            boxes: Boxes in Cartesian coordinates [N, 4]
            
        Returns:
            polar_boxes: Boxes in polar coordinates [N, 4]
        """
        # This is a simplified conversion, assuming image center as origin
        # In a real implementation, you would use the actual image dimensions
        
        # Get center of image (assume normalized coordinates)
        center_x, center_y = 0.5, 0.5
        
        # Extract coordinates
        x1, y1, x2, y2 = boxes.unbind(-1)
        
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
        
        # Stack to get polar boxes
        polar_boxes = torch.stack([r1, theta1, r2, theta2], dim=-1)
        
        return polar_boxes
    
    def _polar_to_cartesian(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from polar [r1, theta1, r2, theta2] to Cartesian [x1, y1, x2, y2] coordinates.
        
        Args:
            boxes: Boxes in polar coordinates [N, 4]
            
        Returns:
            cart_boxes: Boxes in Cartesian coordinates [N, 4]
        """
        # This is a simplified conversion, assuming image center as origin
        # In a real implementation, you would use the actual image dimensions
        
        # Get center of image (assume normalized coordinates)
        center_x, center_y = 0.5, 0.5
        
        # Extract coordinates
        r1, theta1, r2, theta2 = boxes.unbind(-1)
        
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
        
        # Stack to get Cartesian boxes
        cart_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        
        return cart_boxes
    
    def _cartesian_to_polar_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Convert landmarks from Cartesian [x1, y1, x2, y2, ...] to polar [r1, theta1, r2, theta2, ...] coordinates.
        
        Args:
            landmarks: Landmarks in Cartesian coordinates [N, num_landmarks*2]
            
        Returns:
            polar_landmarks: Landmarks in polar coordinates [N, num_landmarks*2]
        """
        # This is a simplified conversion, assuming image center as origin
        # In a real implementation, you would use the actual image dimensions
        
        # Get center of image (assume normalized coordinates)
        center_x, center_y = 0.5, 0.5
        
        # Initialize output tensor
        polar_landmarks = torch.zeros_like(landmarks)
        
        # Convert each landmark point
        for i in range(0, landmarks.shape[1], 2):
            # Get x, y coordinates
            x = landmarks[:, i]
            y = landmarks[:, i+1]
            
            # Skip invisible landmarks (x=0, y=0)
            mask = ~((x == 0) & (y == 0))
            
            # Calculate polar coordinates
            dx = x - center_x
            dy = y - center_y
            
            # Calculate radius and angle
            r = torch.sqrt(dx**2 + dy**2)
            theta = torch.atan2(dy, dx)
            
            # Set invisible landmarks to (0, 0) in polar coordinates
            polar_landmarks[:, i] = r * mask
            polar_landmarks[:, i+1] = theta * mask
        
        return polar_landmarks
    
    def _polar_to_cartesian_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Convert landmarks from polar [r1, theta1, r2, theta2, ...] to Cartesian [x1, y1, x2, y2, ...] coordinates.
        
        Args:
            landmarks: Landmarks in polar coordinates [N, num_landmarks*2]
            
        Returns:
            cart_landmarks: Landmarks in Cartesian coordinates [N, num_landmarks*2]
        """
        # This is a simplified conversion, assuming image center as origin
        # In a real implementation, you would use the actual image dimensions
        
        # Get center of image (assume normalized coordinates)
        center_x, center_y = 0.5, 0.5
        
        # Initialize output tensor
        cart_landmarks = torch.zeros_like(landmarks)
        
        # Convert each landmark point
        for i in range(0, landmarks.shape[1], 2):
            # Get r, theta coordinates
            r = landmarks[:, i]
            theta = landmarks[:, i+1]
            
            # Skip invisible landmarks (r=0)
            mask = (r != 0)
            
            # Calculate Cartesian coordinates
            x = center_x + r * torch.cos(theta)
            y = center_y + r * torch.sin(theta)
            
            # Set invisible landmarks to (0, 0) in Cartesian coordinates
            cart_landmarks[:, i] = x * mask
            cart_landmarks[:, i+1] = y * mask
        
        return cart_landmarks
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1: First set of boxes [N, 4]
            boxes2: Second set of boxes [M, 4]
            
        Returns:
            iou: IoU between boxes [N, M]
            union: Union area between boxes [N, M]
        """
        area1 = torch.prod(boxes1[:, 2:] - boxes1[:, :2], 1)
        area2 = torch.prod(boxes2[:, 2:] - boxes2[:, :2], 1)
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        
        union = area1[:, None] + area2 - inter
        
        iou = inter / union
        
        return iou, union
    
    def _generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute generalized IoU between two sets of boxes.
        
        Args:
            boxes1: First set of boxes [N, 4]
            boxes2: Second set of boxes [M, 4]
            
        Returns:
            giou: Generalized IoU between boxes [N, M]
        """
        # Calculate IoU
        iou, union = self._box_iou(boxes1, boxes2)
        
        # Calculate enclosing box
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        area = wh[:, :, 0] * wh[:, :, 1]
        
        # Calculate GIoU
        giou = iou - (area - union) / area.clamp(min=1e-6)
        
        return giou
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the criterion.
        
        Args:
            outputs: Dictionary with model outputs
            targets: List of dictionaries with targets
            
        Returns:
            losses: Dictionary with losses
        """
        # Get outputs from last layer
        if 'aux_outputs' in outputs:
            out = outputs['aux_outputs'][-1]
            out.update({k: v for k, v in outputs.items() if k != 'aux_outputs'})
            outputs = out
        
        # Retrieve the matching between outputs and targets
        indices = self.matcher(outputs, targets)
        
        # Compute the average number of target boxes across all batches
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Reduce across all processes for distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else num_boxes, min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f'loss_{loss}')(outputs, targets, indices, num_boxes))
        
        # Apply weights to losses
        return {k: v * self.weight_dict.get(k, 1.0) for k, v in losses.items()}
