"""
Miscellaneous utility functions for Polar-RTDETRv2.

This module provides various utility functions for training and evaluation,
including distributed training utilities, logging, and metrics.
"""

import os
import sys
import time
import math
import json
import datetime
import logging
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Iterator

import torch
import torch.nn as nn
import torch.distributed as dist
import subprocess  # Needed for SLURM hostname lookup in init_distributed_mode


def init_distributed_mode(args):
    """
    Initialize distributed training.
    
    Args:
        args: Arguments with distributed training configuration
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.
    
    Returns:
        available: Whether distributed training is available and initialized
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Get the number of processes in the distributed training.
    
    Returns:
        world_size: Number of processes
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process in the distributed training.
    
    Returns:
        rank: Rank of current process
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Check if the current process is the main process.
    
    Returns:
        is_main: Whether current process is the main process
    """
    return get_rank() == 0


def save_config(args, output_dir):
    """
    Save configuration to a JSON file.
    
    Args:
        args: Arguments to save
        output_dir: Output directory
    """
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        # -------- Robust conversion of any `args`/`config` object to dict --------
        # 1. Already a mapping -> use directly
        if isinstance(args, dict):
            config_dict = args
        else:
            # 2. Try the common `vars()` path (works for Namespace / dataclass etc.)
            try:
                config_dict = vars(args)
            except TypeError:
                # 3. Fallback to __dict__ if present
                config_dict = getattr(args, "__dict__", {})
                # 4. Last resort—store string repr to avoid crashing
                if not config_dict:
                    config_dict = {"config": str(args)}
        
        # Remove non-serializable values
        config_dict = {k: v for k, v in config_dict.items() 
                      if not k.startswith('__') and not callable(v)}
        
        # Convert torch tensors to lists
        for k, v in config_dict.items():
            if isinstance(v, torch.Tensor):
                config_dict[k] = v.tolist()
        
        # Save to file
        config_file = os.path.join(output_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)


def setup_for_distributed(is_master):
    """
    Set up process for distributed training.
    
    Args:
        is_master: Whether current process is the master process
    """
    # This function disables printing when not in master process
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    """
    Set up logger for training and testing.
    
    Args:
        name: Logger name
        save_dir: Directory to save log file
        distributed_rank: Process rank
        filename: Log file name
        
    Returns:
        logger: Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Don't log results for non-master processes
    if distributed_rank > 0:
        return logger
    
    # Create handlers
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if save_dir is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_dir, filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def reduce_dict(input_dict, average=True):
    """
    Reduce dictionary values across all processes in distributed training.
    
    Args:
        input_dict: Dictionary to reduce
        average: Whether to average or sum the values
        
    Returns:
        reduced_dict: Reduced dictionary
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        
        # Sort the keys for deterministic behavior
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    Create a learning rate scheduler with linear warmup.
    
    Args:
        optimizer: Optimizer
        warmup_iters: Number of warmup iterations
        warmup_factor: Warmup factor
        
    Returns:
        scheduler: Learning rate scheduler
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        """
        Initialize SmoothedValue.
        
        Args:
            window_size: Window size for smoothing
            fmt: Format string for printing
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Update with a new value.
        
        Args:
            value: New value
            n: Number of items this value represents
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronize values between processes in distributed training.
        """
        if not is_dist_avail_and_initialized():
            return
        
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """Get median value."""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """Get average value over window."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """Get global average value."""
        return self.total / self.count if self.count > 0 else 0

    @property
    def max(self):
        """Get maximum value."""
        return max(self.deque) if len(self.deque) > 0 else 0

    @property
    def value(self):
        """Get last value."""
        return self.deque[-1] if len(self.deque) > 0 else 0

    def __str__(self):
        """Convert to string."""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger:
    """
    Logger for metrics during training and evaluation.
    """
    def __init__(self, delimiter="\t"):
        """
        Initialize MetricLogger.
        
        Args:
            delimiter: Delimiter for printing
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Update metrics.
        
        Args:
            **kwargs: Metrics to update
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """Get attribute."""
        if attr in self.meters:
            return self.meters[attr]
        return getattr(self, attr)

    def __str__(self):
        """Convert to string."""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        Synchronize metrics between processes in distributed training.
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        Add a meter.
        
        Args:
            name: Meter name
            meter: Meter instance
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Log metrics every print_freq iterations.
        
        Args:
            iterable: Iterable to iterate over
            print_freq: Frequency of printing
            header: Header string
            
        Returns:
            iterable: Iterator over iterable
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            
            i += 1
            end = time.time()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    """
    Collate function for data loader.
    
    Args:
        batch: Batch of examples
        
    Returns:
        images: Batch of images
        targets: Batch of targets
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, targets
