"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Training script for Polar Face Detection and Landmark Localization
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


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
    
    print("✅ Configuration validation passed")


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
    
    if writer and dist_utils.is_main_process():
        writer.add_scalar('Model/TotalParams', total_params, 0)
        writer.add_scalar('Model/TrainableParams', trainable_params, 0)


def log_training_progress(epoch, stats, writer=None):
    """Log training progress"""
    if writer and dist_utils.is_main_process():
        for key, value in stats.items():
            if 'loss' in key.lower():
                writer.add_scalar(f'Train/{key}', value, epoch)
            elif key == 'lr':
                writer.add_scalar('Learning_Rate/lr', value, epoch)
            else:
                writer.add_scalar(f'Train/{key}', value, epoch)


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


def main(args):
    """Main training function"""
    
    # Setup distributed training
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    
    # Load and validate configuration
    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() 
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    validate_config(cfg)
    
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
    
    # Create solver and setup training
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.train()  # This sets up the model, datasets, etc.
    
    # Log model information
    log_model_info(solver.model, writer)
    
    # Training loop
    print("🚀 Starting polar face training...")
    print(f"📊 Training setup:")
    print(f"   - Epochs: {cfg.epoches}")
    print(f"   - Batch size: {cfg.train_dataloader.batch_size}")
    print(f"   - Learning rate: {cfg.optimizer.param_groups[0]['lr']}")
    print(f"   - Device: {solver.device}")
    print(f"   - Output directory: {output_dir}")
    
    if args.test_only:
        print("🧪 Running validation only...")
        solver.val()
        return
    
    best_ap = 0.0
    start_time = time.time()
    
    for epoch in range(solver.last_epoch + 1, cfg.epoches):
        print(f"\n📅 Epoch {epoch}/{cfg.epoches}")
        
        # Set epoch for distributed training
        if hasattr(solver.train_dataloader, 'sampler') and hasattr(solver.train_dataloader.sampler, 'set_epoch'):
            solver.train_dataloader.sampler.set_epoch(epoch)
        
        if hasattr(solver.train_dataloader, 'set_epoch'):
            solver.train_dataloader.set_epoch(epoch)
        
        # Training
        epoch_start = time.time()
        train_stats = train_one_epoch_polar_face(
            solver, epoch, writer, args.print_freq
        )
        epoch_time = time.time() - epoch_start
        
        # Learning rate scheduling
        if solver.lr_warmup_scheduler is None or solver.lr_warmup_scheduler.finished():
            solver.lr_scheduler.step()
        
        solver.last_epoch = epoch
        
        # Validation
        val_stats, coco_evaluator = evaluate_polar_face(solver)
        
        # Get AP for best model tracking
        current_ap = 0.0
        if coco_evaluator and 'bbox' in coco_evaluator.coco_eval:
            current_ap = coco_evaluator.coco_eval['bbox'].stats[0]  # AP@0.5:0.95
        
        is_best = current_ap > best_ap
        if is_best:
            best_ap = current_ap
        
        # Logging
        log_training_progress(epoch, train_stats, writer)
        if writer and dist_utils.is_main_process():
            for key, value in val_stats.items():
                if 'coco_eval' in key and isinstance(value, list):
                    for i, v in enumerate(value):
                        writer.add_scalar(f'Val/{key}_{i}', v, epoch)
                else:
                    writer.add_scalar(f'Val/{key}', value, epoch)
            
            writer.add_scalar('Val/AP', current_ap, epoch)
            writer.add_scalar('Time/EpochTime', epoch_time, epoch)
        
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
                'epoch_time': epoch_time
            }
            
            with (output_dir / "training_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    # Training completed
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\n🎉 Training completed!')
    print(f'⏱️  Total training time: {total_time_str}')
    print(f'🏆 Best AP achieved: {best_ap:.4f}')
    
    if writer:
        writer.close()
    
    dist_utils.cleanup()


def train_one_epoch_polar_face(solver, epoch, writer=None, print_freq=50):
    """Train one epoch with polar face specific logging"""
    from src.solver.det_engine import train_one_epoch
    
    # Call the standard training function
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
        writer=writer
    )
    
    return stats


def evaluate_polar_face(solver):
    """Evaluate with polar face specific metrics"""
    from src.solver.det_engine import evaluate
    
    # Use EMA model if available
    module = solver.ema.module if solver.ema else solver.model
    
    stats, coco_evaluator = evaluate(
        module,
        solver.criterion,
        solver.postprocessor,
        solver.val_dataloader,
        solver.evaluator,
        solver.device
    )
    
    return stats, coco_evaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Polar Face Training')
    
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