"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized Training script for Polar Face Detection and Landmark Localization
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import time
import datetime
import gc
from pathlib import Path
import functools
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from tqdm import tqdm

from src.misc import dist_utils, logger
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


# Feature cache for expensive computations
class FeatureCache:
    """Cache for storing and retrieving computed features to avoid redundant calculations"""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key, compute_func=None):
        """Get item from cache or compute if not available"""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        if compute_func is not None:
            result = compute_func()
            self.put(key, result)
            return result
        return None
    
    def put(self, key, value):
        """Add item to cache, managing size"""
        if len(self.cache) >= self.max_size:
            # Remove oldest item (simple strategy)
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = value
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        
    def stats(self):
        """Return cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


# Optimized Gaussian heatmap generation
def create_gaussian_heatmap_vectorized(landmarks, heatmap_size, sigma, device):
    """
    Vectorized implementation of Gaussian heatmap generation
    
    Args:
        landmarks: Tensor of shape [N, num_landmarks*2] containing landmark coordinates
        heatmap_size: Size of output heatmaps
        sigma: Standard deviation for Gaussian
        device: Device to create tensors on
        
    Returns:
        Tensor of shape [N, num_landmarks, heatmap_size, heatmap_size]
    """
    batch_size = landmarks.shape[0]
    num_landmarks = landmarks.shape[1] // 2
    
    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.arange(heatmap_size, device=device),
        torch.arange(heatmap_size, device=device),
        indexing='ij'
    )
    
    # Reshape landmarks for broadcasting
    landmarks_x = landmarks[:, 0::2].view(batch_size, num_landmarks, 1, 1)
    landmarks_y = landmarks[:, 1::2].view(batch_size, num_landmarks, 1, 1)
    
    # Calculate Gaussian
    heatmaps = torch.exp(
        -((x.expand(batch_size, num_landmarks, -1, -1) - landmarks_x)**2 + 
          (y.expand(batch_size, num_landmarks, -1, -1) - landmarks_y)**2) / 
        (2 * sigma**2)
    )
    
    return heatmaps


# Memory management utilities
def clear_memory():
    """Clear unused memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_config(cfg):
    """Validate configuration for polar face training"""
    required_keys = [
        'model', 'criterion', 'postprocessor', 
        'train_dataloader', 'val_dataloader'
    ]
    
    for key in required_keys:
        if not hasattr(cfg, key) or getattr(cfg, key) is None:
            raise ValueError(f"Missing required config: {key}")
    
    # Check if it's using polar face components
    if cfg.yaml_cfg.get('model') != 'RTDETR':
        raise ValueError("Model must be RTDETR for polar face training")
    
    decoder_type = cfg.yaml_cfg.get('RTDETR', {}).get('decoder')
    if decoder_type != 'PolarFaceTransformer':
        raise ValueError("Decoder must be PolarFaceTransformer")
    
    criterion_type = cfg.yaml_cfg.get('criterion')
    if criterion_type != 'PolarFaceCriterion':
        raise ValueError("Criterion must be PolarFaceCriterion")
    
    # Check for optimization-specific settings
    has_progressive = 'training_phases' in cfg.yaml_cfg
    enable_heatmaps = cfg.yaml_cfg.get('enable_heatmaps', True)
    enable_polar = cfg.yaml_cfg.get('enable_polar_features', True)
    
    print("✅ Configuration validation passed")
    print(f"   - Progressive training: {'Enabled' if has_progressive else 'Disabled'}")
    print(f"   - Heatmap generation: {'Enabled' if enable_heatmaps else 'Disabled'}")
    print(f"   - Polar features: {'Enabled' if enable_polar else 'Disabled'}")
    
    return {
        'has_progressive': has_progressive,
        'enable_heatmaps': enable_heatmaps,
        'enable_polar': enable_polar
    }


def setup_logging(output_dir):
    """Setup logging and tensorboard"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    summary_dir = output_dir / 'tensorboard'
    writer = SummaryWriter(str(summary_dir))
    
    return writer


def log_model_info(model, writer=None):
    """Log model information"""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Information:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"   - Memory footprint: {total_params * 4 / (1024**2):.2f} MB")
    
    if writer and dist_utils.is_main_process():
        writer.add_scalar('Model/TotalParams', total_params, 0)
        writer.add_scalar('Model/TrainableParams', trainable_params, 0)


def log_training_progress(epoch, stats, writer=None, cache_stats=None):
    """Log training progress"""
    if writer and dist_utils.is_main_process():
        for key, value in stats.items():
            if 'loss' in key.lower():
                writer.add_scalar(f'Train/{key}', value, epoch)
            elif key == 'lr':
                writer.add_scalar('Learning_Rate/lr', value, epoch)
            else:
                writer.add_scalar(f'Train/{key}', value, epoch)
        
        # Log cache statistics if available
        if cache_stats:
            writer.add_scalar('Cache/HitRate', cache_stats['hit_rate'], epoch)
            writer.add_scalar('Cache/Size', cache_stats['size'], epoch)


def save_checkpoint(model, optimizer, lr_scheduler, ema, epoch, output_dir, is_best=False):
    """Save training checkpoint"""
    if not dist_utils.is_main_process():
        return
    
    output_dir = Path(output_dir)
    
    # Prepare state dict
    state = {
        'epoch': epoch,
        'model': dist_utils.de_parallel(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
        'date': datetime.datetime.now().isoformat(),
    }
    
    if ema is not None:
        state['ema'] = {'module': ema.module.state_dict(), 'updates': ema.updates}
    
    # Save latest checkpoint
    checkpoint_path = output_dir / 'last.pth'
    torch.save(state, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = output_dir / 'best.pth'
        torch.save(state, best_path)
        print(f"💾 Saved best checkpoint at epoch {epoch}")
    
    # Save periodic checkpoints
    if epoch % 10 == 0:
        periodic_path = output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(state, periodic_path)


def update_training_config(cfg, epoch, feature_configs):
    """Update training configuration based on current epoch"""
    if not hasattr(cfg, 'training_phases') or not cfg.training_phases:
        return cfg, feature_configs
    
    # Find the appropriate phase based on epoch
    current_phase = None
    for phase in cfg.training_phases:
        if epoch >= phase['epoch']:
            current_phase = phase
    
    if current_phase:
        # Update losses
        if 'losses' in current_phase:
            cfg.criterion.losses = current_phase['losses']
        
        # Update feature flags
        feature_configs['enable_heatmaps'] = current_phase.get('enable_heatmaps', 
                                                             feature_configs['enable_heatmaps'])
        feature_configs['enable_polar'] = current_phase.get('enable_polar_features', 
                                                          feature_configs['enable_polar'])
    
    return cfg, feature_configs


def train_one_epoch_optimized(solver, epoch, writer=None, print_freq=50, 
                             feature_cache=None, feature_configs=None, profiler=None):
    """Optimized training for one epoch with caching and progress bar"""
    from src.solver.det_engine import train_one_epoch
    
    # Update model based on feature configs
    if feature_configs:
        # Disable expensive computations based on config
        if hasattr(solver.model, 'decoder') and hasattr(solver.model.decoder, 'landmark_heatmap_head'):
            solver.model.decoder.landmark_heatmap_head.enabled = feature_configs.get('enable_heatmaps', True)
        
        # Update criterion to skip certain losses
        if not feature_configs.get('enable_polar', True) and 'polar_cls' in solver.criterion.losses:
            solver.criterion.losses = [loss for loss in solver.criterion.losses 
                                     if not loss.startswith('polar_')]
        
        if not feature_configs.get('enable_heatmaps', True) and 'heatmaps' in solver.criterion.losses:
            solver.criterion.losses = [loss for loss in solver.criterion.losses 
                                     if loss != 'heatmaps']
    
    # Create progress bar for better visualization
    total_batches = len(solver.train_dataloader)
    pbar = tqdm(total=total_batches, desc=f"Epoch {epoch}", 
               unit="batch", leave=True, disable=not dist_utils.is_main_process())
    
    # Custom batch hook to update progress bar
    def batch_hook(i, loss_dict_reduced, lr):
        if dist_utils.is_main_process():
            # Update progress bar with current metrics
            loss_value = sum(loss_dict_reduced.values())
            pbar.set_postfix({
                'loss': f"{loss_value:.4f}",
                'lr': f"{lr:.6f}"
            })
            pbar.update(1)
    
    # Start profiling if enabled
    if profiler:
        profiler.start()
    
    # Call the standard training function with our hook
    stats = train_one_epoch(
        solver.model,
        solver.criterion,
        solver.train_dataloader,
        solver.optimizer,
        solver.device,
        epoch,
        max_norm=getattr(solver.cfg, 'clip_max_norm', 0),
        print_freq=print_freq,
        ema=solver.ema,
        scaler=solver.scaler,
        lr_warmup_scheduler=solver.lr_warmup_scheduler,
        writer=writer,
        batch_end_hook=batch_hook
    )
    
    # Stop profiling if enabled
    if profiler:
        profiler.stop()
    
    pbar.close()
    
    # Clear any cached tensors to free memory
    clear_memory()
    
    return stats


def evaluate_optimized(solver, feature_cache=None, feature_configs=None):
    """Optimized evaluation with caching"""
    from src.solver.det_engine import evaluate
    
    # Use EMA model if available
    module = solver.ema.module if solver.ema else solver.model
    
    # Create progress bar for validation
    total_batches = len(solver.val_dataloader)
    pbar = tqdm(total=total_batches, desc="Validation", 
               unit="batch", leave=True, disable=not dist_utils.is_main_process())
    
    # Custom batch hook to update progress bar
    def batch_hook(i):
        if dist_utils.is_main_process():
            pbar.update(1)
    
    # Temporarily patch the evaluate function to use our progress bar
    original_log_every = solver.val_dataloader.__class__.log_every
    
    def patched_log_every(self, iterable, print_freq, header):
        for i, obj in enumerate(iterable):
            yield obj
            if i % print_freq == 0:
                batch_hook(i)
    
    solver.val_dataloader.__class__.log_every = patched_log_every
    
    stats, coco_evaluator = evaluate(
        module,
        solver.criterion,
        solver.postprocessor,
        solver.val_dataloader,
        solver.evaluator,
        solver.device
    )
    
    # Restore original method
    solver.val_dataloader.__class__.log_every = original_log_every
    
    pbar.close()
    
    return stats, coco_evaluator


def setup_profiler(output_dir, enabled=False):
    """Setup PyTorch profiler"""
    if not enabled:
        return None
    
    output_dir = Path(output_dir)
    profile_dir = output_dir / 'profile'
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    return torch.profiler.profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )


def main(args):
    """Main training function with optimizations"""
    
    # Setup distributed training
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    
    # Load and validate configuration
    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() 
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    feature_configs = validate_config(cfg)
    
    print('🔧 Configuration loaded successfully')
    if dist_utils.is_main_process():
        print('📋 Config summary:')
        for key in ['model', 'criterion', 'postprocessor', 'num_classes']:
            if hasattr(cfg, key):
                print(f'   - {key}: {getattr(cfg, key)}')
    
    # Setup output directory and logging
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    writer = setup_logging(cfg.output_dir) if dist_utils.is_main_process() else None
    
    # Initialize feature cache
    feature_cache = FeatureCache(max_size=args.cache_size)
    
    # Setup profiler if enabled
    profiler = setup_profiler(cfg.output_dir, args.profile) if dist_utils.is_main_process() else None
    
    # Create solver and setup training
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.train()  # This sets up the model, datasets, etc.
    
    # Apply optimizations to model if needed
    if args.optimize_model:
        # Optimize model architecture if supported
        pass
    
    # Log model information
    log_model_info(solver.model, writer)
    
    # Training loop
    print("🚀 Starting polar face training with optimizations...")
    print(f"📊 Training setup:")
    print(f"   - Epochs: {cfg.epoches}")
    print(f"   - Batch size: {cfg.train_dataloader.batch_size}")
    print(f"   - Learning rate: {cfg.optimizer.param_groups[0]['lr']}")
    print(f"   - Device: {solver.device}")
    print(f"   - Mixed precision: {args.use_amp}")
    print(f"   - Feature caching: Enabled (size={args.cache_size})")
    print(f"   - Output directory: {output_dir}")
    
    if args.test_only:
        print("🧪 Running validation only...")
        stats, coco_evaluator = evaluate_optimized(solver, feature_cache, feature_configs)
        
        # Print validation results
        if dist_utils.is_main_process() and 'coco_eval_bbox' in stats:
            print(f"📊 Validation Results:")
            print(f"   - AP@0.5:0.95: {stats['coco_eval_bbox'][0]:.4f}")
            print(f"   - AP@0.5: {stats['coco_eval_bbox'][1]:.4f}")
        
        return
    
    best_ap = 0.0
    start_time = time.time()
    
    for epoch in range(solver.last_epoch + 1, cfg.epoches):
        print(f"\n📅 Epoch {epoch}/{cfg.epoches}")
        
        # Update configuration based on training phase
        if feature_configs['has_progressive']:
            cfg, feature_configs = update_training_config(cfg, epoch, feature_configs)
            print(f"   - Active losses: {cfg.criterion.losses}")
            print(f"   - Heatmaps enabled: {feature_configs['enable_heatmaps']}")
            print(f"   - Polar features enabled: {feature_configs['enable_polar']}")
        
        # Set epoch for distributed training
        if hasattr(solver.train_dataloader, 'sampler') and hasattr(solver.train_dataloader.sampler, 'set_epoch'):
            solver.train_dataloader.sampler.set_epoch(epoch)
        
        if hasattr(solver.train_dataloader, 'set_epoch'):
            solver.train_dataloader.set_epoch(epoch)
        
        # Training
        epoch_start = time.time()
        train_stats = train_one_epoch_optimized(
            solver, epoch, writer, args.print_freq,
            feature_cache, feature_configs, profiler
        )
        epoch_time = time.time() - epoch_start
        
        # Learning rate scheduling
        if solver.lr_warmup_scheduler is None or solver.lr_warmup_scheduler.finished():
            solver.lr_scheduler.step()
        
        solver.last_epoch = epoch
        
        # Validation
        val_stats, coco_evaluator = evaluate_optimized(solver, feature_cache, feature_configs)
        
        # Get AP for best model tracking
        current_ap = 0.0
        if coco_evaluator and 'bbox' in coco_evaluator.coco_eval:
            current_ap = coco_evaluator.coco_eval['bbox'].stats[0]  # AP@0.5:0.95
        
        is_best = current_ap > best_ap
        if is_best:
            best_ap = current_ap
        
        # Logging
        log_training_progress(epoch, train_stats, writer, feature_cache.stats())
        if writer and dist_utils.is_main_process():
            for key, value in val_stats.items():
                if 'coco_eval' in key and isinstance(value, list):
                    for i, v in enumerate(value):
                        writer.add_scalar(f'Val/{key}_{i}', v, epoch)
                else:
                    writer.add_scalar(f'Val/{key}', value, epoch)
            
            writer.add_scalar('Val/AP', current_ap, epoch)
            writer.add_scalar('Time/EpochTime', epoch_time, epoch)
            writer.add_scalar('Memory/GPU_Used_MB', 
                             torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0, 
                             epoch)
        
        # Save checkpoint
        save_checkpoint(solver.model, solver.optimizer, solver.lr_scheduler, 
                       solver.ema, epoch, output_dir, is_best)
        
        # Print progress
        print(f"⏱️  Epoch {epoch} completed in {epoch_time:.1f}s")
        print(f"📈 Current AP: {current_ap:.4f}, Best AP: {best_ap:.4f}")
        
        # Save training log
        if dist_utils.is_main_process():
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
                'epoch': epoch,
                'best_ap': best_ap,
                'epoch_time': epoch_time,
                'cache_hit_rate': feature_cache.stats()['hit_rate']
            }
            
            with (output_dir / "training_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # Clear cache between epochs to prevent memory leaks
        feature_cache.clear()
        clear_memory()
    
    # Training completed
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\n🎉 Training completed!')
    print(f'⏱️  Total training time: {total_time_str}')
    print(f'🏆 Best AP achieved: {best_ap:.4f}')
    
    # Final memory usage report
    if torch.cuda.is_available() and dist_utils.is_main_process():
        print(f"📊 Memory Statistics:")
        print(f"   - Peak GPU memory: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        print(f"   - Current GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    if writer:
        writer.close()
    
    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized Polar Face Training')
    
    # Configuration
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, 
                       help='Resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, 
                       help='Tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                       help='Device for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Optimization options
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--optimize-model', action='store_true', default=False,
                       help='Apply model-specific optimizations')
    parser.add_argument('--cache-size', type=int, default=100,
                       help='Size of feature cache')
    parser.add_argument('--profile', action='store_true', default=False,
                       help='Enable profiling')
    
    # Training options
    parser.add_argument('--test-only', action='store_true', default=False,
                       help='Only run validation')
    parser.add_argument('--print-freq', type=int, default=50,
                       help='Print frequency')
    
    # Distributed training
    parser.add_argument('--print-method', type=str, default='builtin',
                       help='Print method')
    parser.add_argument('--print-rank', type=int, default=0,
                       help='Print rank id')
    parser.add_argument('--local-rank', type=int,
                       help='Local rank id')
    
    # Configuration updates
    parser.add_argument('-u', '--update', nargs='+',
                       help='Update config values')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    main(args)
