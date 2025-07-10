"""
Matcher module for Polar-RTDETRv2.

This module provides the Hungarian matcher implementation for Polar-RTDETRv2,
which matches predicted boxes and landmarks to ground truth targets.
"""

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for Polar-RTDETRv2.
    
    This class implements the Hungarian bipartite matching algorithm to find the optimal
    assignment between predicted boxes/landmarks and ground truth targets.
    
    The cost function considers:
    - Classification cost: -log(p) where p is the predicted probability for the correct class
    - Box cost: L1 distance between predicted and ground truth boxes
    - GIoU cost: Generalized IoU loss between predicted and ground truth boxes
    - Landmark cost: L1 distance between predicted and ground truth landmarks (if available)
    
    The matcher supports both Cartesian and polar coordinate representations.
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 1.0,
        cost_giou: float = 1.0,
        cost_landmarks: float = 1.0,
        use_polar: bool = False
    ):
        """
        Initialize the matcher.
        
        Args:
            cost_class: Weight for classification cost
            cost_bbox: Weight for L1 box coordinate cost
            cost_giou: Weight for generalized IoU cost
            cost_landmarks: Weight for L1 landmark coordinate cost
            use_polar: Whether to use polar coordinate representation
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_landmarks = cost_landmarks
        self.use_polar = use_polar
        
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_landmarks != 0, \
            "At least one cost should be non-zero"
    
    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the assignment between predictions and targets.
        
        Args:
            outputs: Dictionary with model outputs:
                - 'pred_logits': Classification logits [batch_size, num_queries, num_classes]
                - 'pred_boxes': Predicted boxes [batch_size, num_queries, 4]
                - 'pred_landmarks' (optional): Predicted landmarks [batch_size, num_queries, num_landmarks*2]
            targets: List of dictionaries with targets:
                - 'labels': Target class labels [num_targets]
                - 'boxes': Target boxes [num_targets, 4]
                - 'landmarks' (optional): Target landmarks [num_targets, num_landmarks*2]
                - 'use_polar': Whether targets are in polar coordinates
        
        Returns:
            List of tuples (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Check if we need to convert coordinates
        use_polar_targets = any(v.get("use_polar", False) for v in targets)
        
        # Convert coordinates if needed
        if self.use_polar != use_polar_targets:
            if self.use_polar:
                # Convert target boxes from Cartesian to polar
                tgt_bbox = self._cartesian_to_polar(tgt_bbox)
            else:
                # Convert target boxes from polar to Cartesian
                tgt_bbox = self._polar_to_cartesian(tgt_bbox)
        
        # Compute the classification cost
        # Negative log probability of the correct class
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the giou cost between boxes
        if self.use_polar:
            # For polar coordinates, convert to Cartesian for IoU calculation
            out_bbox_cart = self._polar_to_cartesian(out_bbox)
            tgt_bbox_cart = self._polar_to_cartesian(tgt_bbox) if use_polar_targets else tgt_bbox
            cost_giou = -self._generalized_box_iou(out_bbox_cart, tgt_bbox_cart)
        else:
            cost_giou = -self._generalized_box_iou(out_bbox, tgt_bbox)
        
        # Final cost matrix
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        
        # Add landmark cost if available
        if "pred_landmarks" in outputs and all("landmarks" in v for v in targets):
            out_landmarks = outputs["pred_landmarks"].flatten(0, 1)  # [batch_size * num_queries, num_landmarks*2]
            tgt_landmarks = torch.cat([v["landmarks"] for v in targets])
            
            # Convert coordinates if needed
            if self.use_polar != use_polar_targets:
                if self.use_polar:
                    # Convert target landmarks from Cartesian to polar
                    tgt_landmarks = self._cartesian_to_polar_landmarks(tgt_landmarks)
                else:
                    # Convert target landmarks from polar to Cartesian
                    tgt_landmarks = self._polar_to_cartesian_landmarks(tgt_landmarks)
            
            # Compute the L1 cost between landmarks
            # Only consider visible landmarks (non-zero)
            landmark_cost = torch.zeros_like(cost_bbox)
            
            # Calculate landmark cost for each pair
            for i in range(out_landmarks.shape[0]):
                for j in range(tgt_landmarks.shape[0]):
                    # Get landmarks for this pair
                    pred_lm = out_landmarks[i]
                    tgt_lm = tgt_landmarks[j]
                    
                    # Create visibility mask (1 for visible landmarks, 0 for invisible)
                    if self.use_polar:
                        # In polar coordinates, radius=0 means invisible
                        vis_mask = torch.ones_like(tgt_lm)
                        for k in range(0, tgt_lm.shape[0], 2):
                            if tgt_lm[k] == 0:  # radius = 0 means invisible
                                vis_mask[k] = 0
                                vis_mask[k+1] = 0
                    else:
                        # In Cartesian coordinates, x=0 and y=0 means invisible
                        vis_mask = torch.ones_like(tgt_lm)
                        for k in range(0, tgt_lm.shape[0], 2):
                            if tgt_lm[k] == 0 and tgt_lm[k+1] == 0:
                                vis_mask[k] = 0
                                vis_mask[k+1] = 0
                    
                    # Calculate L1 distance only for visible landmarks
                    if vis_mask.sum() > 0:
                        l1_dist = torch.abs(pred_lm - tgt_lm) * vis_mask
                        landmark_cost[i, j] = l1_dist.sum() / (vis_mask.sum() + 1e-8)
                    else:
                        landmark_cost[i, j] = 0
            
            # Add landmark cost to final cost matrix
            C += self.cost_landmarks * landmark_cost
        
        # Reshape cost matrix to account for batch dimension
        C = C.view(bs, num_queries, -1).cpu()
        
        # Get number of targets per batch element
        sizes = [len(v["boxes"]) for v in targets]
        
        # Perform Hungarian matching
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            # Get indices of optimal assignment
            indices_i, indices_j = linear_sum_assignment(c[i])
            indices.append((torch.as_tensor(indices_i, dtype=torch.int64), 
                           torch.as_tensor(indices_j, dtype=torch.int64)))
        
        return indices
    
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


def build_matcher(args):
    """
    Build matcher from arguments.
    
    Args:
        args: Arguments with matcher configuration
        
    Returns:
        matcher: HungarianMatcher instance
    """
    # Get matcher parameters from args
    cost_class = args.set_cost_class if hasattr(args, 'set_cost_class') else 1.0
    cost_bbox = args.set_cost_bbox if hasattr(args, 'set_cost_bbox') else 1.0
    cost_giou = args.set_cost_giou if hasattr(args, 'set_cost_giou') else 1.0
    cost_landmarks = args.set_cost_landmarks if hasattr(args, 'set_cost_landmarks') else 1.0
    use_polar = args.use_polar if hasattr(args, 'use_polar') else False
    
    # Create matcher
    matcher = HungarianMatcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
        cost_landmarks=cost_landmarks,
        use_polar=use_polar
    )
    
    return matcher
