"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Polar Face Postprocessor for face detection, landmark localization, and polar domain features
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import numpy as np

from ...core import register


__all__ = ['PolarFacePostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PolarFacePostProcessor(nn.Module):
    """Postprocessor for polar face transformer outputs"""
    
    __share__ = [
        'num_classes', 
        'use_focal_loss', 
        'num_top_queries', 
        'remap_mscoco_category'
    ]
    
    def __init__(
        self, 
        num_classes=1, 
        use_focal_loss=True, 
        num_top_queries=300, 
        remap_mscoco_category=False,
        num_landmarks=5,
        polar_bins=36,
        landmark_threshold=0.5,
        nms_threshold=0.5
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False
        
        # Face-specific parameters
        self.num_landmarks = num_landmarks
        self.polar_bins = polar_bins
        self.landmark_threshold = landmark_threshold
        self.nms_threshold = nms_threshold

    def extra_repr(self) -> str:
        return (f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, '
                f'num_top_queries={self.num_top_queries}, num_landmarks={self.num_landmarks}, '
                f'polar_bins={self.polar_bins}')
    
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        """
        Args:
            outputs: Dict containing model predictions
            orig_target_sizes: [B, 2] tensor with original image sizes (w, h)
        """
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        
        # Convert boxes to xyxy format and scale to original image size
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # Process landmarks if available
        landmarks = None
        if 'pred_landmarks' in outputs:
            pred_landmarks = outputs['pred_landmarks']
            
            if self.use_focal_loss:
                landmarks = pred_landmarks.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, pred_landmarks.shape[-1]))
            else:
                if pred_landmarks.shape[1] > self.num_top_queries:
                    landmarks = torch.gather(pred_landmarks, dim=1, 
                                           index=index.unsqueeze(-1).tile(1, 1, pred_landmarks.shape[-1]))
                else:
                    landmarks = pred_landmarks
            
            # Scale landmarks to original image size
            landmarks = self._scale_landmarks(landmarks, orig_target_sizes)

        # Process polar domain features if available
        polar_features = None
        if 'pred_polar' in outputs:
            polar_pred = outputs['pred_polar']
            polar_features = {}
            
            # Polar angle classification
            if 'polar_angle_logits' in polar_pred:
                polar_angle_logits = polar_pred['polar_angle_logits']
                if self.use_focal_loss:
                    polar_angle_logits = polar_angle_logits.gather(dim=1, 
                        index=index.unsqueeze(-1).repeat(1, 1, polar_angle_logits.shape[-1]))
                
                polar_angle_probs = F.softmax(polar_angle_logits, dim=-1)
                polar_angle_pred = torch.argmax(polar_angle_probs, dim=-1)
                polar_features['angle_bin'] = polar_angle_pred
                polar_features['angle_confidence'] = torch.max(polar_angle_probs, dim=-1)[0]
            
            # Polar angle regression
            if 'polar_angle_reg' in polar_pred:
                polar_angle_reg = polar_pred['polar_angle_reg']
                if self.use_focal_loss:
                    polar_angle_reg = polar_angle_reg.gather(dim=1, 
                        index=index.unsqueeze(-1).repeat(1, 1, polar_angle_reg.shape[-1]))
                polar_features['angle_offset'] = polar_angle_reg.squeeze(-1)
            
            # Polar magnitude
            if 'polar_magnitude' in polar_pred:
                polar_magnitude = polar_pred['polar_magnitude']
                if self.use_focal_loss:
                    polar_magnitude = polar_magnitude.gather(dim=1, 
                        index=index.unsqueeze(-1).repeat(1, 1, polar_magnitude.shape[-1]))
                polar_features['magnitude'] = polar_magnitude.squeeze(-1)
            
            # Convert to actual angles
            if 'angle_bin' in polar_features and 'angle_offset' in polar_features:
                bin_size = 2 * np.pi / self.polar_bins
                bin_centers = (polar_features['angle_bin'].float() + 0.5) * bin_size
                angle_offsets = polar_features['angle_offset'] * bin_size
                polar_features['angle_radians'] = bin_centers + angle_offsets
                polar_features['angle_degrees'] = polar_features['angle_radians'] * 180.0 / np.pi

        # Process landmark heatmaps if available
        landmark_heatmaps = None
        if 'landmark_heatmaps' in outputs:
            landmark_heatmaps = outputs['landmark_heatmaps']
            # Convert heatmaps to landmark coordinates
            heatmap_landmarks = self._heatmaps_to_landmarks(landmark_heatmaps, orig_target_sizes)

        # TODO for onnx export
        if self.deploy_mode:
            result = {
                'labels': labels, 
                'boxes': boxes, 
                'scores': scores
            }
            if landmarks is not None:
                result['landmarks'] = landmarks
            if polar_features is not None:
                result.update(polar_features)
            return result

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for i, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            
            # Add landmarks if available
            if landmarks is not None:
                result['landmarks'] = landmarks[i]
            
            # Add polar features if available
            if polar_features is not None:
                result['polar_features'] = {k: v[i] for k, v in polar_features.items()}
            
            # Add heatmap landmarks if available
            if landmark_heatmaps is not None:
                result['heatmap_landmarks'] = heatmap_landmarks[i]
                
            results.append(result)
        
        return results

    def _scale_landmarks(self, landmarks, orig_target_sizes):
        """Scale normalized landmarks to original image size"""
        # landmarks: [B, N, num_landmarks*2]
        # orig_target_sizes: [B, 2] (w, h)
        
        batch_size, num_queries, landmark_dim = landmarks.shape
        landmarks = landmarks.reshape(batch_size, num_queries, self.num_landmarks, 2)
        
        # Scale landmarks
        orig_sizes_expanded = orig_target_sizes.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 2]
        landmarks = landmarks * orig_sizes_expanded
        
        return landmarks.reshape(batch_size, num_queries, landmark_dim)

    def _heatmaps_to_landmarks(self, heatmaps, orig_target_sizes):
        """Convert heatmaps to landmark coordinates"""
        # heatmaps: [B, num_landmarks, H, W]
        # orig_target_sizes: [B, 2] (w, h)
        
        batch_size, num_landmarks, heatmap_h, heatmap_w = heatmaps.shape
        landmarks = []
        
        for b in range(batch_size):
            batch_landmarks = []
            orig_w, orig_h = orig_target_sizes[b]
            
            for lm in range(num_landmarks):
                heatmap = heatmaps[b, lm]
                
                # Find peak location
                max_val, max_idx = torch.max(heatmap.flatten(), dim=0)
                
                if max_val > self.landmark_threshold:
                    max_y = max_idx // heatmap_w
                    max_x = max_idx % heatmap_w
                    
                    # Convert to original image coordinates
                    x = (max_x.float() / heatmap_w) * orig_w
                    y = (max_y.float() / heatmap_h) * orig_h
                    
                    batch_landmarks.extend([x.item(), y.item()])
                else:
                    # If no confident peak, use center of heatmap
                    batch_landmarks.extend([orig_w/2, orig_h/2])
            
            landmarks.append(batch_landmarks)
        
        return torch.tensor(landmarks, device=heatmaps.device)

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self