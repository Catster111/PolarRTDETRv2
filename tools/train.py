#!/usr/bin/env python
"""
Training script for Polar-RTDETRv2.

This script handles the full training pipeline for Polar-RTDETRv2 model
with WiderFace dataset including 5 face landmarks. It supports:
- Distributed training
- Mixed precision (Automatic Mixed Precision)
- Checkpointing and resuming
- Tensorboard and WandB logging
- Configuration via YAML files
"""

import os
import sys
import time
import datetime
import json
import random
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Prefer the newer torch.amp API (PyTorch ≥ 2.0).  Fall back to the older
# torch.cuda.amp for compatibility with earlier versions.
try:
    from torch.amp import GradScaler, autocast
    _USE_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _USE_NEW_AMP = False

import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# NOTE: use the project-local dataset utilities, **not** Hugging-Face `datasets`
# to avoid the name-clash that caused the previous ImportError.
from polar_datasets import build_widerface, data_prefetcher
from models import build_model
from models.polar_rtdetrv2 import PolarRTDETRv2
from engine.matcher import build_matcher
from engine.criterion import SetCriterion
from utils.box_ops import box_iou, generalized_box_iou
from utils.misc import (
    MetricLogger, SmoothedValue, reduce_dict, warmup_lr_scheduler,
    setup_logger, save_config, is_main_process, get_rank, get_world_size,
    init_distributed_mode, setup_for_distributed
)
from utils.visualizer import Visualizer


logger = logging.getLogger(__name__)


def get_args_parser():
    """
    Parse command line arguments.
    
    Returns:
        parser: ArgumentParser object
    """
    parser = argparse.ArgumentParser('Polar-RTDETRv2 Training', add_help=False)
    
    # File paths
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Path to output directory')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Load from pretrained checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size per GPU (overrides config)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of data loading workers (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay (overrides config)')
    
    # Distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl',
                        help='Distributed backend')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    
    # Mixed precision and performance
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--no-amp', action='store_false', dest='amp',
                        help='Disable Automatic Mixed Precision')
    parser.set_defaults(amp=True)
    
    # Logging and evaluation
    parser.add_argument('--eval-freq', type=int, default=None,
                        help='Evaluation frequency in epochs (overrides config)')
    parser.add_argument('--save-freq', type=int, default=None,
                        help='Checkpoint saving frequency in epochs (overrides config)')
    parser.add_argument('--log-freq', type=int, default=None,
                        help='Logging frequency in iterations (overrides config)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--no-wandb', action='store_false', dest='wandb',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization during evaluation')
    
    # Debug and development
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic mode')
    
    return parser


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config_with_args(config, args):
    """
    Override config values with command line arguments.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        config: Updated configuration dictionary
    """
    # Training parameters
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['dataset']['batch_size'] = args.batch_size
    if args.workers is not None:
        config['dataset']['num_workers'] = args.workers
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay
    
    # Evaluation and logging
    if args.eval_freq is not None:
        config['evaluation']['eval_freq'] = args.eval_freq
    if args.save_freq is not None:
        config['evaluation']['save_freq'] = args.save_freq
    if args.log_freq is not None:
        config['logging']['log_freq'] = args.log_freq
    if args.wandb is not None:
        config['logging']['wandb'] = args.wandb
    
    # Debug
    if args.debug:
        config['logging']['debug']['enabled'] = True
    
    return config


def setup_wandb(config, args):
    """
    Set up Weights & Biases logging.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    if not is_main_process() or not config['logging'].get('wandb', False):
        return
    
    try:
        import wandb
        
        # Initialize wandb
        wandb.init(
            project="polar-rtdetrv2",
            name=f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            dir=args.output_dir
        )
        
        logger.info("Weights & Biases logging enabled")
    except ImportError:
        logger.warning("wandb package not found. Weights & Biases logging disabled.")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")


def set_random_seed(seed, deterministic=False):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def build_optimizer(model, config):
    """
    Build optimizer for training.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        optimizer: PyTorch optimizer
    """
    params = []
    
    # Separate backbone parameters for different learning rate
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
    
    # Add parameter groups with different learning rates
    params.append({
        'params': backbone_params,
        'lr': config['training']['lr_backbone']
    })
    params.append({
        'params': other_params,
        'lr': config['training']['lr']
    })
    
    # Create optimizer
    optimizer_name = config['training']['optimizer']['name'].lower()
    
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            params,
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            weight_decay=config['training']['weight_decay'],
            betas=(
                config['training']['optimizer'].get('beta1', 0.9),
                config['training']['optimizer'].get('beta2', 0.999)
            )
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def build_lr_scheduler(optimizer, config, len_data_loader):
    """
    Build learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
        len_data_loader: Length of data loader
        
    Returns:
        lr_scheduler: PyTorch learning rate scheduler
    """
    scheduler_name = config['training']['lr_scheduler']['name'].lower()
    epochs = config['training']['epochs']
    
    if scheduler_name == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['lr_drop'],
            gamma=config['training']['lr_drop_factor']
        )
    elif scheduler_name == 'multistep':
        milestones = config['training']['lr_scheduler'].get('milestones', [int(epochs * 0.8)])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=config['training']['lr_drop_factor']
        )
    elif scheduler_name == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    else:
        raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")
    
    # Add warmup if enabled
    if config['training']['lr_scheduler'].get('warmup', False):
        warmup_epochs = config['training']['lr_scheduler'].get('warmup_epochs', 5)
        warmup_factor = config['training']['lr_scheduler'].get('warmup_factor', 0.1)
        warmup_iters = len_data_loader * warmup_epochs
        
        # Create warmup scheduler that will be called before the main scheduler
        warmup_scheduler = warmup_lr_scheduler(
            optimizer,
            warmup_iters,
            warmup_factor
        )
        
        return warmup_scheduler, lr_scheduler
    
    return None, lr_scheduler


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    config,
    scaler=None,
    warmup_scheduler=None,
    max_norm=None,
    log_freq=10
):
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        criterion: Loss function
        data_loader: Data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch
        config: Configuration dictionary
        scaler: Gradient scaler for mixed precision
        warmup_scheduler: Learning rate warmup scheduler
        max_norm: Maximum norm for gradient clipping
        log_freq: Logging frequency
        
    Returns:
        metric_logger: Metric logger with training statistics
    """
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    
    # Use data prefetcher for faster data loading
    prefetcher = data_prefetcher(data_loader, device)
    samples, targets = prefetcher.next()
    
    i = 0
    start_time = time.time()
    
    # Training loop
    while samples is not None:
        i += 1

        # Forward pass with mixed precision if enabled
        if scaler is not None:
            # Use the appropriate autocast context based on available API
            if _USE_NEW_AMP:
                with autocast(device_type=device.type):
                    outputs = model(samples)
                    loss_dict = criterion(outputs, targets)
                    weight_dict = criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            else:
                with autocast():
                    outputs = model(samples)
                    loss_dict = criterion(outputs, targets)
                    weight_dict = criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        else:
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Reduce losses over all GPUs
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(losses).backward()
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        
        # -----------------------------------------------------------------
        # Warm-up LR update *after* the optimizer.step(), as required by
        # PyTorch (avoids skipping the first LR value and silences warning).
        # -----------------------------------------------------------------
        if warmup_scheduler is not None and epoch == 0:
            warmup_scheduler.step()
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        
        # Log metrics
        metric_logger.update(loss=losses_reduced_scaled.item())
        for k, v in loss_dict_reduced_scaled.items():
            metric_logger.update(**{k: v.item()})
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        
        # Log to wandb
        if is_main_process() and config['logging'].get('wandb', False) and i % log_freq == 0:
            try:
                import wandb
                wandb.log({
                    'train/loss': losses_reduced_scaled.item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    **{f'train/{k}': v.item() for k, v in loss_dict_reduced_scaled.items()},
                    'epoch': epoch,
                    'iter': i + epoch * len(data_loader)
                })
            except:
                pass
        
        # Get next batch
        samples, targets = prefetcher.next()
        
        # Print progress
        if i % log_freq == 0 or samples is None:
            eta_seconds = metric_logger.meters['loss'].global_avg * (len(data_loader) - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            logger.info(
                metric_logger.delimiter.join([
                    header,
                    f"[{i}/{len(data_loader)}]",
                    f'eta: {eta_string}',
                    f'time: {time.time() - start_time:.4f}',
                    f"{str(metric_logger)}"
                ])
            )
            
            start_time = time.time()
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    
    return metric_logger


@torch.no_grad()
def evaluate(
    model,
    criterion,
    data_loader,
    device,
    epoch,
    config,
    visualizer=None
):
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        criterion: Loss function
        data_loader: Data loader
        device: Device to evaluate on
        epoch: Current epoch
        config: Configuration dictionary
        visualizer: Visualization helper
        
    Returns:
        stats: Evaluation statistics
    """
    model.eval()
    criterion.eval()
    
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # Initialize metrics
    all_preds = []
    all_targets = []
    
    # Use data prefetcher for faster data loading
    prefetcher = data_prefetcher(data_loader, device)
    samples, targets = prefetcher.next()
    
    i = 0
    start_time = time.time()
    
    # Evaluation loop
    while samples is not None:
        i += 1
        
        # Forward pass
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # Reduce losses over all GPUs
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_value = sum(loss_dict_reduced_scaled.values())
        
        # Update metrics
        metric_logger.update(loss=loss_value)
        for k, v in loss_dict_reduced_scaled.items():
            metric_logger.update(**{k: v.item()})
        
        # Process predictions for metrics
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # Process landmark predictions if available
        if 'pred_landmarks' in outputs:
            pred_landmarks = outputs['pred_landmarks']
        else:
            pred_landmarks = None
        
        # Convert predictions to format suitable for metrics
        for idx, (logits, boxes, target) in enumerate(zip(pred_logits, pred_boxes, targets)):
            # Get probabilities and class predictions
            prob = F.softmax(logits, dim=-1)
            scores, labels = prob[..., 1].max(dim=0)  # Only consider face class (1)
            
            # Keep predictions above threshold
            keep = scores > config['inference']['confidence_threshold']
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            
            # Convert to CPU for metrics
            pred_dict = {
                'scores': scores.cpu(),
                'labels': labels.cpu(),
                'boxes': boxes.cpu()
            }
            
            # Add landmarks if available
            if pred_landmarks is not None:
                landmarks = pred_landmarks[idx][keep]
                pred_dict['landmarks'] = landmarks.cpu()
            
            all_preds.append(pred_dict)
            all_targets.append({
                'boxes': target['boxes'].cpu(),
                'labels': target['labels'].cpu(),
                'image_id': target['image_id'],
                'orig_size': target['orig_size'].cpu()
            })
            
            # Add landmarks to targets if available
            if 'landmarks' in target:
                all_targets[-1]['landmarks'] = target['landmarks'].cpu()
            
            # Visualize predictions
            if visualizer is not None and is_main_process() and i % 10 == 0:
                visualizer.visualize_detection(
                    samples[idx].cpu(),
                    pred_dict,
                    target,
                    epoch,
                    idx + i * len(samples)
                )
        
        # Get next batch
        samples, targets = prefetcher.next()
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # Compute metrics
    stats = compute_detection_metrics(all_preds, all_targets, config)
    
    # Add loss metrics
    stats.update({f'loss/{k}': v.global_avg for k, v in metric_logger.meters.items()})
    
    # Log to wandb
    if is_main_process() and config['logging'].get('wandb', False):
        try:
            import wandb
            wandb.log({
                **{f'val/{k}': v for k, v in stats.items()},
                'epoch': epoch
            })
        except:
            pass
    
    # Print results
    logger.info(f"Evaluation results for epoch {epoch}:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v:.4f}")
    
    return stats


def compute_detection_metrics(all_preds, all_targets, config):
    """
    Compute detection metrics (mAP, precision, recall).
    
    Args:
        all_preds: List of prediction dictionaries
        all_targets: List of target dictionaries
        config: Configuration dictionary
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Initialize metrics
    metrics = {}
    
    # Skip if no predictions
    if len(all_preds) == 0:
        logger.warning("No predictions to evaluate")
        return {'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0}
    
    # Compute AP for each IoU threshold
    iou_thresholds = config['evaluation']['metrics'].get('map_iou_thresholds', [0.5, 0.75])
    
    # Initialize AP metrics
    ap_metrics = {f'AP_{int(iou * 100)}': [] for iou in iou_thresholds}
    ap_metrics['AP'] = []  # Mean across all IoU thresholds
    
    # Compute AP for each image
    for pred, target in zip(all_preds, all_targets):
        # Skip if no predictions or no targets
        if len(pred['boxes']) == 0 or len(target['boxes']) == 0:
            continue
        
        # Compute IoU between predictions and targets
        iou_matrix = box_iou(pred['boxes'], target['boxes'])
        
        # Compute AP for each IoU threshold
        for iou_thresh in iou_thresholds:
            ap = compute_average_precision(
                pred['scores'],
                iou_matrix,
                iou_thresh
            )
            ap_metrics[f'AP_{int(iou_thresh * 100)}'].append(ap)
        
        # Compute mean AP across IoU thresholds
        mean_ap = sum(ap_metrics[f'AP_{int(iou * 100)}'][-1] for iou in iou_thresholds) / len(iou_thresholds)
        ap_metrics['AP'].append(mean_ap)
    
    # Compute mean over all images
    for k, v in ap_metrics.items():
        metrics[f'm{k}'] = sum(v) / max(len(v), 1)
    
    # Compute landmark metrics if available
    if 'landmarks' in all_preds[0] and 'landmarks' in all_targets[0]:
        landmark_metrics = compute_landmark_metrics(all_preds, all_targets)
        metrics.update(landmark_metrics)
    
    return metrics


def compute_average_precision(scores, iou_matrix, iou_threshold):
    """
    Compute Average Precision for a single IoU threshold.
    
    Args:
        scores: Prediction confidence scores
        iou_matrix: IoU matrix between predictions and targets
        iou_threshold: IoU threshold
        
    Returns:
        ap: Average Precision
    """
    # Sort predictions by score
    score_indices = torch.argsort(scores, descending=True)
    iou_matrix = iou_matrix[score_indices]
    
    # Mark predictions as TP or FP
    gt_matched = torch.zeros(iou_matrix.shape[1], dtype=torch.bool)
    tp = torch.zeros(len(score_indices), dtype=torch.bool)
    fp = torch.zeros(len(score_indices), dtype=torch.bool)
    
    for i in range(len(score_indices)):
        # Find best matching ground truth
        max_iou, max_idx = torch.max(iou_matrix[i], dim=0)
        
        # If IoU > threshold and ground truth not already matched
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            tp[i] = True
            gt_matched[max_idx] = True
        else:
            fp[i] = True
    
    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    recalls = tp_cumsum / max(gt_matched.sum(), 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Add start and end points
    precisions = torch.cat([torch.tensor([1.0]), precisions])
    recalls = torch.cat([torch.tensor([0.0]), recalls])
    
    # Compute area under PR curve
    ap = torch.trapz(precisions, recalls)
    
    return ap.item()


def compute_landmark_metrics(all_preds, all_targets):
    """
    Compute landmark metrics (NME - Normalized Mean Error).
    
    Args:
        all_preds: List of prediction dictionaries
        all_targets: List of target dictionaries
        
    Returns:
        metrics: Dictionary of landmark metrics
    """
    all_nme = []
    
    for pred, target in zip(all_preds, all_targets):
        # Skip if no predictions or no targets
        if (len(pred['boxes']) == 0 or len(target['boxes']) == 0 or
            'landmarks' not in pred or 'landmarks' not in target):
            continue
        
        # Compute IoU to match predictions with targets
        iou_matrix = box_iou(pred['boxes'], target['boxes'])
        
        # For each target, find best matching prediction
        for i in range(len(target['boxes'])):
            # Find prediction with highest IoU
            max_iou, max_idx = torch.max(iou_matrix[:, i], dim=0)
            
            # Skip if IoU too low
            if max_iou < 0.5:
                continue
            
            # Get landmarks
            pred_lm = pred['landmarks'][max_idx]
            target_lm = target['landmarks'][i]
            
            # Skip if landmarks are not valid
            if torch.all(target_lm == 0):
                continue
            
            # Compute inter-ocular distance for normalization
            # (distance between eyes - landmarks 0 and 1)
            left_eye = target_lm[0:2]
            right_eye = target_lm[2:4]
            norm_factor = torch.sqrt(torch.sum((right_eye - left_eye) ** 2))
            
            # If eyes are not visible, use bbox diagonal
            if norm_factor < 1e-6:
                bbox = target['boxes'][i]
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                norm_factor = torch.sqrt(w ** 2 + h ** 2)
            
            # Compute NME
            nme = 0
            valid_landmarks = 0
            
            for j in range(0, len(target_lm), 2):
                # Skip if landmark is not visible
                if target_lm[j] == 0 and target_lm[j+1] == 0:
                    continue
                
                # Compute Euclidean distance
                dist = torch.sqrt(
                    (pred_lm[j] - target_lm[j]) ** 2 +
                    (pred_lm[j+1] - target_lm[j+1]) ** 2
                )
                
                # Normalize by inter-ocular distance
                nme += dist / norm_factor
                valid_landmarks += 1
            
            # Average NME over valid landmarks
            if valid_landmarks > 0:
                nme /= valid_landmarks
                all_nme.append(nme.item())
    
    # Compute mean NME
    if len(all_nme) > 0:
        mean_nme = sum(all_nme) / len(all_nme)
    else:
        mean_nme = float('inf')
    
    return {'landmark_nme': mean_nme}


def save_checkpoint(model, optimizer, lr_scheduler, scaler, epoch, args, config, stats=None, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        lr_scheduler: Learning rate scheduler to save
        scaler: Gradient scaler to save
        epoch: Current epoch
        args: Command line arguments
        config: Configuration dictionary
        stats: Evaluation statistics
        is_best: Whether this is the best checkpoint
    """
    # Create checkpoint directory
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint
    checkpoint = {
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'args': args,
        'config': config
    }
    
    # Add stats if available
    if stats is not None:
        checkpoint['stats'] = stats
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_{epoch:04d}.pth'
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save as best checkpoint if needed
    if is_best:
        best_path = checkpoint_dir / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")
    
    # Remove old checkpoints if needed
    save_top_k = config['logging']['checkpointing'].get('save_top_k', 3)
    if save_top_k > 0:
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_[0-9]*.pth'))
        if len(checkpoints) > save_top_k:
            for checkpoint_to_remove in checkpoints[:-save_top_k]:
                checkpoint_to_remove.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_to_remove}")


def load_checkpoint(model, optimizer, lr_scheduler, scaler, checkpoint_path):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load
        optimizer: Optimizer to load
        lr_scheduler: Learning rate scheduler to load
        scaler: Gradient scaler to load
        checkpoint_path: Path to checkpoint
        
    Returns:
        epoch: Last epoch
        args: Command line arguments
        config: Configuration dictionary
        stats: Evaluation statistics
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model weights
    model_state_dict = checkpoint['model']
    
    # Handle DataParallel/DDP wrapped models
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    
    # Load optimizer state
    if 'optimizer' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load lr_scheduler state
    if 'lr_scheduler' in checkpoint and lr_scheduler is not None and checkpoint['lr_scheduler'] is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    # Load scaler state
    if 'scaler' in checkpoint and scaler is not None and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    
    # Get epoch, args, config, stats
    epoch = checkpoint.get('epoch', 0)
    args = checkpoint.get('args', None)
    config = checkpoint.get('config', None)
    stats = checkpoint.get('stats', None)
    
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch, args, config, stats


def main(args):
    """
    Main function for training and evaluation.
    
    Args:
        args: Command line arguments
    """
    # Initialize distributed mode
    init_distributed_mode(args)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config = override_config_with_args(config, args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config, output_dir / 'config.yaml')
    
    # Setup logging
    # utils.misc.setup_logger(name, save_dir, distributed_rank, filename)
    setup_logger(
        "train",
        output_dir,
        get_rank(),
        filename="train.log"
    )
    
    # Log basic information
    logger.info(f"Start training with config: {args.config}")
    logger.info(f"World size: {get_world_size()}")
    logger.info(f"Rank: {get_rank()}")
    logger.info(f"Local rank: {args.local_rank}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Set random seed
    set_random_seed(args.seed, args.deterministic)
    
    # Setup wandb
    setup_wandb(config, args)
    
    # Setup device
    # Select proper CUDA device.
    # In non-distributed mode `local_rank` is usually -1; default to GPU 0.
    if torch.cuda.is_available():
        cuda_index = 0 if args.local_rank < 0 else args.local_rank
        device = torch.device(f'cuda:{cuda_index}')
        
        # Log GPU information
        logger.info("=" * 80)
        logger.info("GPU INFORMATION:")
        logger.info(f"  Using CUDA device: {torch.cuda.get_device_name(cuda_index)}")
        logger.info(f"  Device capability: {torch.cuda.get_device_capability(cuda_index)}")
        logger.info(f"  Total GPU memory: {torch.cuda.get_device_properties(cuda_index).total_memory / 1024**3:.2f} GB")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info("=" * 80)
    else:
        device = torch.device('cpu')
        logger.warning("=" * 80)
        logger.warning("WARNING: CUDA is not available. Training will be slow on CPU!")
        logger.warning("=" * 80)
        
    logger.info(f"Using device: {device}")
    
    # ------------------------------------------------------------------
    # Build datasets
    # Validation set is optional – continue gracefully if it's missing.
    # ------------------------------------------------------------------
    logger.info("Building datasets...")
    dataset_train, num_classes = build_widerface('train', config)

    has_val = True
    try:
        dataset_val, _ = build_widerface('val', config)
    except (FileNotFoundError, AssertionError) as e:
        logger.warning(f"Validation set unavailable ({e}); continuing without validation.")
        dataset_val = None
        has_val = False
    
    logger.info(f"Training dataset size: {len(dataset_train)}")
    if has_val:
        logger.info(f"Validation dataset size: {len(dataset_val)}")
    
    # Build data loaders
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False) if has_val else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if has_val else None
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config['dataset']['batch_size'], drop_last=True
    )
    
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=lambda b: tuple(zip(*b)),
        num_workers=config['dataset']['num_workers']
    )
    
    data_loader_val = None
    if has_val:
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=config['dataset']['batch_size'],
            sampler=sampler_val,
            drop_last=False,
            collate_fn=lambda b: tuple(zip(*b)),
            num_workers=config['dataset']['num_workers']
        )
    
    # Build model
    logger.info("Building model...")
    model = build_model(config, num_classes)
    model.to(device)
    
    # Print model summary
    if is_main_process():
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of parameters: {n_parameters}")
    
    # Initialize DDP
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    # Build matcher and criterion
    matcher = build_matcher(config)
    weight_dict = {
        'loss_ce': config['training']['loss_weights']['ce'],
        'loss_bbox': config['training']['loss_weights']['bbox'],
        'loss_giou': config['training']['loss_weights']['giou'],
    }
    
    # Add landmark loss if enabled
    if config['model']['landmarks']['enabled']:
        weight_dict['loss_landmarks'] = config['training']['loss_weights']['landmarks']
    
    # Build criterion
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=['labels', 'boxes', 'landmarks'] if config['model']['landmarks']['enabled'] else ['labels', 'boxes']
    )
    criterion.to(device)
    
    # Build optimizer and lr_scheduler
    optimizer = build_optimizer(model, config)
    warmup_scheduler, lr_scheduler = build_lr_scheduler(optimizer, config, len(data_loader_train))
    
    # Initialize AMP scaler
    if args.amp:
        if _USE_NEW_AMP:
            # For newer PyTorch versions, use the new API
            scaler = GradScaler()
        else:
            # For older PyTorch versions, use the legacy API
            scaler = GradScaler()
    else:
        scaler = None
    
    # Create visualizer
    visualizer = None
    if args.visualize and is_main_process():
        visualizer = Visualizer(output_dir / 'visualizations')
    
    # Load checkpoint if provided
    start_epoch = 0
    best_map = 0.0
    
    if args.resume:
        start_epoch, resume_args, resume_config, resume_stats = load_checkpoint(
            model, optimizer, lr_scheduler, scaler, args.resume
        )
        start_epoch += 1  # Start from next epoch
        
        # Get best mAP
        if resume_stats is not None and 'mAP' in resume_stats:
            best_map = resume_stats['mAP']
    
    # Load pretrained weights if provided
    elif args.pretrained:
        _, _, _, _ = load_checkpoint(model, None, None, None, args.pretrained)
    
    # Training loop
    logger.info("Starting training...")
    epochs = config['training']['epochs']
    
    for epoch in range(start_epoch, epochs):
        # Set epoch for distributed sampler
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        # Train for one epoch
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
            warmup_scheduler=warmup_scheduler if epoch == 0 else None,
            max_norm=config['training'].get('clip_max_norm', None),
            log_freq=config['logging'].get('log_freq', 10)
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        if is_main_process() and (epoch + 1) % config['evaluation'].get('save_freq', 1) == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                epoch=epoch,
                args=args,
                config=config,
                stats=train_stats,
                is_best=False
            )
        
        # Evaluate (only if validation data exists)
        if has_val and (epoch + 1) % config['evaluation'].get('eval_freq', 1) == 0:
            eval_stats = evaluate(
                model=model,
                criterion=criterion,
                data_loader=data_loader_val,
                device=device,
                epoch=epoch,
                config=config,
                visualizer=visualizer
            )
            
            # Save best checkpoint
            if is_main_process() and eval_stats['mAP'] > best_map:
                best_map = eval_stats['mAP']
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    args=args,
                    config=config,
                    stats=eval_stats,
                    is_best=True
                )
    
    # Final evaluation
    if has_val:
        logger.info("Final evaluation...")
        eval_stats = evaluate(
            model=model,
            criterion=criterion,
            data_loader=data_loader_val,
            device=device,
            epoch=epochs,
            config=config,
            visualizer=visualizer
        )
    else:
        eval_stats = {}
    
    # Save final checkpoint
    if is_main_process():
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            epoch=epochs,
            args=args,
            config=config,
            stats=eval_stats if has_val else None,
            is_best=has_val and eval_stats.get('mAP', 0) > best_map
        )
    
    # Log final results
    logger.info("Training completed.")
    if has_val:
        logger.info(f"Best mAP: {best_map:.4f}")
        logger.info(f"Final mAP: {eval_stats['mAP']:.4f}")
    else:
        logger.info("Validation was not performed (no validation set).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Polar-RTDETRv2 Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
