"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Polar Face Criterion for face detection, landmark localization, and polar domain learning
"""

import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision
import numpy as np

import copy

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class PolarFaceCriterion(nn.Module):
    """Multi-task criterion for polar face transformer"""
    
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, 
                 matcher, 
                 weight_dict, 
                 losses, 
                 alpha=0.75, 
                 gamma=2.0, 
                 num_classes=1,
                 num_landmarks=5,
                 polar_bins=36,
                 heatmap_size=64,
                 landmark_sigma=2.0,
                 boxes_weight_format=None,
                 share_matched_indices=False):
        """
        Args:
            matcher: Hungarian matcher
            weight_dict: Loss weights for different tasks
            losses: List of loss types to compute
            alpha, gamma: Focal loss parameters
            num_classes: Number of classes (1 for face)
            num_landmarks: Number of facial landmarks
            polar_bins: Number of polar angle bins
            heatmap_size: Size of landmark heatmaps
            landmark_sigma: Sigma for landmark Gaussian targets
            boxes_weight_format: Format for box weight computation
            share_matched_indices: Whether to share indices across aux losses
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        
        # Face-specific parameters
        self.num_landmarks = num_landmarks
        self.polar_bins = polar_bins
        self.heatmap_size = heatmap_size
        self.landmark_sigma = landmark_sigma

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        """Focal loss for face classification"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        """Varifocal loss for face classification"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Bounding box regression loss"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_landmarks(self, outputs, targets, indices, num_boxes):
        """Landmark regression loss"""
        if 'pred_landmarks' not in outputs:
            return {'loss_landmarks': torch.tensor(0.0, device=outputs['pred_logits'].device)}
        
        idx = self._get_src_permutation_idx(indices)
        src_landmarks = outputs['pred_landmarks'][idx]  # [N, num_landmarks*2]
        
        # Get target landmarks
        target_landmarks_list = []
        for t, (_, i) in zip(targets, indices):
            if 'landmarks' in t and t['landmarks'] is not None:
                target_landmarks_list.append(t['landmarks'][i])
            else:
                # Create dummy landmarks if not available
                dummy_landmarks = torch.zeros(len(i), self.num_landmarks * 2, 
                                            device=src_landmarks.device)
                target_landmarks_list.append(dummy_landmarks)
        
        if target_landmarks_list:
            target_landmarks = torch.cat(target_landmarks_list, dim=0)
        else:
            return {'loss_landmarks': torch.tensor(0.0, device=src_landmarks.device)}

        # Normalize landmarks to [0, 1] based on image size
        # Assuming landmarks are already normalized in the dataset
        loss_landmarks = F.smooth_l1_loss(src_landmarks, target_landmarks, reduction='none')
        loss_landmarks = loss_landmarks.sum() / num_boxes

        return {'loss_landmarks': loss_landmarks}

    def loss_landmark_heatmaps(self, outputs, targets, indices, num_boxes):
        """Landmark heatmap loss"""
        if 'landmark_heatmaps' not in outputs:
            return {'loss_heatmaps': torch.tensor(0.0, device=outputs['pred_logits'].device)}
        
        pred_heatmaps = outputs['landmark_heatmaps']  # [B, num_landmarks, H, W]
        batch_size = pred_heatmaps.shape[0]
        
        # Generate target heatmaps
        target_heatmaps = []
        for i, target in enumerate(targets):
            if 'landmarks' in target and target['landmarks'] is not None:
                # Get image size for this batch item
                img_h, img_w = target.get('orig_size', [640, 640])
                heatmap = self._generate_target_heatmaps(
                    target['landmarks'], img_w, img_h)
                target_heatmaps.append(heatmap)
            else:
                # Create empty heatmaps
                empty_heatmap = torch.zeros(self.num_landmarks, self.heatmap_size, self.heatmap_size,
                                          device=pred_heatmaps.device)
                target_heatmaps.append(empty_heatmap)
        
        target_heatmaps = torch.stack(target_heatmaps, dim=0)
        
        # MSE loss for heatmaps
        loss_heatmaps = F.mse_loss(pred_heatmaps, target_heatmaps, reduction='mean')
        
        return {'loss_heatmaps': loss_heatmaps}

    def loss_polar_classification(self, outputs, targets, indices, num_boxes):
        """Polar angle classification loss"""
        if 'pred_polar' not in outputs or 'polar_angle_logits' not in outputs['pred_polar']:
            return {'loss_polar_cls': torch.tensor(0.0, device=outputs['pred_logits'].device)}
        
        idx = self._get_src_permutation_idx(indices)
        src_polar_logits = outputs['pred_polar']['polar_angle_logits'][idx]
        
        # Get target polar features
        target_polar_bins = []
        for t, (_, i) in zip(targets, indices):
            if 'polar_features' in t:
                polar_bins = t['polar_features']['polar_angle_bin'][i]
                target_polar_bins.append(polar_bins)
            else:
                # Create dummy targets
                dummy_bins = torch.zeros(len(i), dtype=torch.long, device=src_polar_logits.device)
                target_polar_bins.append(dummy_bins)
        
        if target_polar_bins:
            target_polar_bins = torch.cat(target_polar_bins, dim=0)
        else:
            return {'loss_polar_cls': torch.tensor(0.0, device=src_polar_logits.device)}
        
        # Cross entropy loss for polar classification
        loss_polar_cls = F.cross_entropy(src_polar_logits, target_polar_bins, reduction='mean')
        
        return {'loss_polar_cls': loss_polar_cls}

    def loss_polar_regression(self, outputs, targets, indices, num_boxes):
        """Polar angle regression and magnitude loss"""
        if 'pred_polar' not in outputs:
            return {
                'loss_polar_reg': torch.tensor(0.0, device=outputs['pred_logits'].device),
                'loss_polar_mag': torch.tensor(0.0, device=outputs['pred_logits'].device)
            }
        
        idx = self._get_src_permutation_idx(indices)
        
        losses = {}
        
        # Angle regression loss
        if 'polar_angle_reg' in outputs['pred_polar']:
            src_angle_reg = outputs['pred_polar']['polar_angle_reg'][idx].squeeze(-1)
            
            target_angle_regs = []
            for t, (_, i) in zip(targets, indices):
                if 'polar_features' in t:
                    angle_regs = t['polar_features']['polar_angle_reg'][i]
                    target_angle_regs.append(angle_regs)
                else:
                    dummy_regs = torch.zeros(len(i), device=src_angle_reg.device)
                    target_angle_regs.append(dummy_regs)
            
            if target_angle_regs:
                target_angle_regs = torch.cat(target_angle_regs, dim=0)
                loss_polar_reg = F.smooth_l1_loss(src_angle_reg, target_angle_regs, reduction='mean')
            else:
                loss_polar_reg = torch.tensor(0.0, device=src_angle_reg.device)
            
            losses['loss_polar_reg'] = loss_polar_reg
        
        # Magnitude loss
        if 'polar_magnitude' in outputs['pred_polar']:
            src_magnitude = outputs['pred_polar']['polar_magnitude'][idx].squeeze(-1)
            
            target_magnitudes = []
            for t, (_, i) in zip(targets, indices):
                if 'polar_features' in t:
                    magnitudes = t['polar_features']['polar_magnitude'][i]
                    target_magnitudes.append(magnitudes)
                else:
                    dummy_mags = torch.ones(len(i), device=src_magnitude.device)
                    target_magnitudes.append(dummy_mags)
            
            if target_magnitudes:
                target_magnitudes = torch.cat(target_magnitudes, dim=0)
                loss_polar_mag = F.smooth_l1_loss(src_magnitude, target_magnitudes, reduction='mean')
            else:
                loss_polar_mag = torch.tensor(0.0, device=src_magnitude.device)
            
            losses['loss_polar_mag'] = loss_polar_mag
        
        return losses

    def _generate_target_heatmaps(self, landmarks, img_w, img_h):
        """Generate target heatmaps from landmark coordinates"""
        num_faces = landmarks.shape[0]
        heatmaps = torch.zeros(num_faces, self.num_landmarks, self.heatmap_size, self.heatmap_size,
                              device=landmarks.device)
        
        scale_x = self.heatmap_size / img_w
        scale_y = self.heatmap_size / img_h
        
        for face_idx in range(num_faces):
            face_landmarks = landmarks[face_idx]  # [num_landmarks * 2]
            
            for lm_idx in range(self.num_landmarks):
                x = face_landmarks[lm_idx * 2] * scale_x
                y = face_landmarks[lm_idx * 2 + 1] * scale_y
                
                # Create Gaussian heatmap
                heatmap = self._create_gaussian_heatmap(
                    x, y, self.heatmap_size, self.heatmap_size, self.landmark_sigma)
                heatmaps[face_idx, lm_idx] = heatmap
        
        # Average over faces if multiple faces
        if num_faces > 1:
            heatmaps = heatmaps.mean(dim=0, keepdim=True)
        
        return heatmaps.squeeze(0)  # Remove face dimension for single face

    def _create_gaussian_heatmap(self, center_x, center_y, width, height, sigma):
        """Create a Gaussian heatmap centered at (center_x, center_y)"""
        x = torch.arange(0, width, dtype=torch.float32, device=center_x.device)
        y = torch.arange(0, height, dtype=torch.float32, device=center_y.device)
        
        y = y.unsqueeze(1)
        
        # Gaussian formula
        heatmap = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
        
        return heatmap

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'landmarks': self.loss_landmarks,
            'heatmaps': self.loss_landmark_heatmaps,
            'polar_cls': self.loss_polar_classification,
            'polar_reg': self.loss_polar_regression,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation."""
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes across all nodes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)            
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            
            # Handle multi-loss returns (like polar_regression)
            if isinstance(l_dict, dict):
                for k, v in l_dict.items():
                    if k in self.weight_dict:
                        losses[k] = v * self.weight_dict[k]
                    else:
                        losses[k] = v
            else:
                # Single loss case
                l_dict = {loss: l_dict}
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    
                    # Handle multi-loss returns
                    if isinstance(l_dict, dict):
                        for k, v in l_dict.items():
                            if k in self.weight_dict:
                                losses[f'{k}_aux_{i}'] = v * self.weight_dict[k]
                            else:
                                losses[f'{k}_aux_{i}'] = v
                    else:
                        l_dict = {loss: l_dict}
                        l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                        l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        # In case of cdn auxiliary losses. For denoising
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    
                    # Handle multi-loss returns
                    if isinstance(l_dict, dict):
                        for k, v in l_dict.items():
                            if k in self.weight_dict:
                                losses[f'{k}_dn_{i}'] = v * self.weight_dict[k]
                            else:
                                losses[f'{k}_dn_{i}'] = v
                    else:
                        l_dict = {loss: l_dict}
                        l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                        l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        # In case of encoder auxiliary losses
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                for loss in self.losses:
                    # Skip face-specific losses for encoder outputs
                    if loss in ['landmarks', 'heatmaps', 'polar_cls', 'polar_reg']:
                        continue
                        
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    
                    # Handle multi-loss returns
                    if isinstance(l_dict, dict):
                        for k, v in l_dict.items():
                            if k in self.weight_dict:
                                losses[f'{k}_enc_{i}'] = v * self.weight_dict[k]
                            else:
                                losses[f'{k}_enc_{i}'] = v
                    else:
                        l_dict = {loss: l_dict}
                        l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                        l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
            
            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        """Get additional meta information for loss computation"""
        if self.boxes_weight_format is None:
            return {}

        # Only compute box weights for box-related losses
        if loss not in ['boxes', 'vfl']:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            return {}

        if loss == 'boxes':
            meta = {'boxes_weight': iou}
        elif loss == 'vfl':
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices for denoising"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.long, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.long, device=device), 
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices