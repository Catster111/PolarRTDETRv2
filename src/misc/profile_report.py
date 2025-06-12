"""
Performance profiling and analysis utilities for PolarRTDETRv2.

This module provides tools to analyze PyTorch profiling data, identify bottlenecks,
and generate optimization recommendations specifically for polar face models.
"""

import os
import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
import torch.nn.functional as F

# Set plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


class ProfileAnalyzer:
    """Analyzer for PyTorch profiler data with focus on PolarRTDETRv2 models."""
    
    # Known bottleneck operations in polar face model
    POLAR_BOTTLENECKS = {
        'gaussian_heatmap': [
            '_create_gaussian_heatmap', 
            '_generate_landmark_heatmaps',
            'LandmarkHeatmapHead'
        ],
        'polar_features': [
            '_generate_polar_features',
            'PolarEmbedding',
            'polar_angle'
        ],
        'transformer_attention': [
            'MultiheadAttention',
            'MSDeformableAttention',
            'cross_attn',
            'self_attn'
        ],
        'loss_computation': [
            'loss_landmark',
            'loss_polar',
            'loss_heatmaps',
            'HungarianMatcher'
        ]
    }
    
    def __init__(self, profile_path: Union[str, Path], model=None):
        """
        Initialize the profile analyzer.
        
        Args:
            profile_path: Path to the profiler output directory or file
            model: Optional model instance for memory analysis
        """
        self.profile_path = Path(profile_path)
        self.model = model
        self.trace_data = None
        self.events_df = None
        self.module_stats = None
        self.memory_stats = None
        self.bottlenecks = []
        
        if self.profile_path.exists():
            self._load_profile_data()
    
    def _load_profile_data(self):
        """Load profiler data from files."""
        if self.profile_path.is_file() and self.profile_path.suffix == '.json':
            with open(self.profile_path, 'r') as f:
                self.trace_data = json.load(f)
        elif (self.profile_path / 'trace.json').exists():
            with open(self.profile_path / 'trace.json', 'r') as f:
                self.trace_data = json.load(f)
        else:
            raise FileNotFoundError(f"No profile data found at {self.profile_path}")
        
        # Extract events
        if self.trace_data and 'traceEvents' in self.trace_data:
            self._process_trace_events()
    
    def _process_trace_events(self):
        """Process trace events into a pandas DataFrame."""
        events = []
        
        for event in self.trace_data['traceEvents']:
            if 'cat' in event and 'name' in event and 'dur' in event:
                events.append({
                    'name': event['name'],
                    'category': event['cat'],
                    'duration': event['dur'] / 1000,  # Convert to ms
                    'ts': event['ts'],
                    'pid': event.get('pid', 0),
                    'tid': event.get('tid', 0),
                    'args': event.get('args', {})
                })
        
        if events:
            self.events_df = pd.DataFrame(events)
            # Extract module name from event name
            self.events_df['module'] = self.events_df['name'].apply(
                lambda x: x.split('::')[0] if '::' in x else x)
    
    def analyze_tensor_operations(self):
        """
        Analyze tensor operations to identify inefficient patterns.
        
        Returns:
            DataFrame with tensor operation statistics
        """
        if self.events_df is None:
            raise ValueError("No profile data loaded")
        
        # Filter for tensor operations
        tensor_ops = self.events_df[
            self.events_df['name'].str.contains('aten::') | 
            self.events_df['name'].str.contains('cudnn::') |
            self.events_df['name'].str.contains('cuda::')
        ].copy()
        
        if tensor_ops.empty:
            print("No tensor operations found in profile data")
            return pd.DataFrame()
        
        # Group by operation name
        op_stats = tensor_ops.groupby('name').agg({
            'duration': ['count', 'sum', 'mean', 'max'],
            'args': lambda x: list(x)
        }).reset_index()
        
        # Flatten the column hierarchy
        op_stats.columns = ['_'.join(col).strip('_') for col in op_stats.columns.values]
        
        # Sort by total duration
        op_stats = op_stats.sort_values('duration_sum', ascending=False)
        
        # Analyze shapes and dtypes where available
        op_stats['shapes'] = op_stats['args_<lambda>'].apply(
            lambda args_list: self._extract_shapes(args_list)
        )
        
        op_stats['dtypes'] = op_stats['args_<lambda>'].apply(
            lambda args_list: self._extract_dtypes(args_list)
        )
        
        # Add percentage of total time
        total_time = op_stats['duration_sum'].sum()
        op_stats['percentage'] = op_stats['duration_sum'] / total_time * 100
        
        # Add cumulative percentage
        op_stats['cumulative_percentage'] = op_stats['percentage'].cumsum()
        
        # Identify potential issues
        op_stats['issues'] = op_stats.apply(self._identify_tensor_issues, axis=1)
        
        # Select and rename columns for better readability
        result = op_stats[[
            'name_', 'duration_count', 'duration_sum', 'duration_mean',
            'duration_max', 'percentage', 'cumulative_percentage', 
            'shapes', 'dtypes', 'issues'
        ]].rename(columns={
            'name_': 'operation',
            'duration_count': 'calls',
            'duration_sum': 'total_ms',
            'duration_mean': 'mean_ms',
            'duration_max': 'max_ms'
        })
        
        return result
    
    def _extract_shapes(self, args_list):
        """Extract tensor shapes from args."""
        shapes = []
        for args in args_list:
            if isinstance(args, dict) and 'input_shapes' in args:
                shapes.append(args['input_shapes'])
        
        # Return most common shapes
        if shapes:
            counter = Counter([str(s) for s in shapes])
            return [f"{shape} ({count}x)" for shape, count in counter.most_common(3)]
        return []
    
    def _extract_dtypes(self, args_list):
        """Extract tensor dtypes from args."""
        dtypes = []
        for args in args_list:
            if isinstance(args, dict) and 'dtype' in args:
                dtypes.append(args['dtype'])
        
        # Return most common dtypes
        if dtypes:
            counter = Counter(dtypes)
            return [f"{dtype} ({count}x)" for dtype, count in counter.most_common(3)]
        return []
    
    def _identify_tensor_issues(self, row):
        """Identify potential issues with tensor operations."""
        issues = []
        op_name = row['name_']
        
        # Check for type conversions
        if 'to(' in op_name or 'type_as' in op_name or 'cast' in op_name:
            issues.append("Type conversion")
        
        # Check for memory-intensive operations
        if 'cat(' in op_name or 'stack(' in op_name or 'concat(' in op_name:
            issues.append("Memory-intensive concatenation")
        
        # Check for inefficient reshaping
        if 'view(' in op_name or 'reshape(' in op_name:
            if row['duration_count'] > 100:
                issues.append("Frequent reshaping")
        
        # Check for expensive operations
        if 'exp(' in op_name or 'pow(' in op_name or 'sqrt(' in op_name:
            if row['duration_sum'] > 50:  # ms
                issues.append("Expensive math operation")
        
        # Check for potential mixed precision issues
        dtypes = row['dtypes']
        if dtypes and any('float32' in d for d in dtypes) and row['percentage'] > 5:
            issues.append("Consider mixed precision")
        
        return issues if issues else None
    
    def analyze_memory_usage(self):
        """
        Analyze GPU memory usage and identify memory bottlenecks.
        
        Returns:
            DataFrame with memory usage statistics
        """
        if self.events_df is None:
            raise ValueError("No profile data loaded")
        
        # Filter for memory allocation events
        memory_events = self.events_df[
            self.events_df['name'].str.contains('malloc') | 
            self.events_df['name'].str.contains('free') |
            self.events_df['name'].str.contains('allocator')
        ].copy()
        
        if memory_events.empty:
            print("No memory events found in profile data")
            return pd.DataFrame()
        
        # Extract memory allocation sizes
        memory_events['bytes'] = memory_events['args'].apply(
            lambda args: args.get('size', 0) if isinstance(args, dict) else 0
        )
        
        # Convert to MB for readability
        memory_events['size_mb'] = memory_events['bytes'] / (1024 * 1024)
        
        # Group by event name
        mem_stats = memory_events.groupby('name').agg({
            'bytes': ['count', 'sum', 'mean', 'max'],
            'size_mb': ['sum', 'mean', 'max']
        }).reset_index()
        
        # Flatten the column hierarchy
        mem_stats.columns = ['_'.join(col).strip('_') for col in mem_stats.columns.values]
        
        # Sort by total allocation
        mem_stats = mem_stats.sort_values('bytes_sum', ascending=False)
        
        # Calculate total allocated memory
        total_allocated = mem_stats[mem_stats['name_'].str.contains('malloc')]['bytes_sum'].sum()
        
        # Add percentage of total allocation
        mem_stats['percentage'] = mem_stats['bytes_sum'] / total_allocated * 100 if total_allocated > 0 else 0
        
        # Select and rename columns for better readability
        result = mem_stats[[
            'name_', 'bytes_count', 'size_mb_sum', 'size_mb_mean',
            'size_mb_max', 'percentage'
        ]].rename(columns={
            'name_': 'operation',
            'bytes_count': 'allocations',
            'size_mb_sum': 'total_mb',
            'size_mb_mean': 'mean_mb',
            'size_mb_max': 'max_mb'
        })
        
        self.memory_stats = result
        return result
    
    def analyze_modules(self):
        """
        Analyze time spent in each module.
        
        Returns:
            DataFrame with module statistics
        """
        if self.events_df is None:
            raise ValueError("No profile data loaded")
        
        # Group by module name
        module_stats = self.events_df.groupby('module').agg({
            'duration': ['count', 'sum', 'mean', 'max']
        }).reset_index()
        
        # Flatten the column hierarchy
        module_stats.columns = ['_'.join(col).strip('_') for col in module_stats.columns.values]
        
        # Sort by total duration
        module_stats = module_stats.sort_values('duration_sum', ascending=False)
        
        # Add percentage of total time
        total_time = module_stats['duration_sum'].sum()
        module_stats['percentage'] = module_stats['duration_sum'] / total_time * 100
        
        # Add cumulative percentage
        module_stats['cumulative_percentage'] = module_stats['percentage'].cumsum()
        
        # Identify polar-specific modules
        module_stats['category'] = module_stats['module_'].apply(self._categorize_module)
        
        # Select and rename columns for better readability
        result = module_stats[[
            'module_', 'duration_count', 'duration_sum', 'duration_mean',
            'duration_max', 'percentage', 'cumulative_percentage', 'category'
        ]].rename(columns={
            'module_': 'module',
            'duration_count': 'calls',
            'duration_sum': 'total_ms',
            'duration_mean': 'mean_ms',
            'duration_max': 'max_ms'
        })
        
        self.module_stats = result
        return result
    
    def _categorize_module(self, module_name):
        """Categorize module based on its name."""
        for category, patterns in self.POLAR_BOTTLENECKS.items():
            if any(pattern in module_name for pattern in patterns):
                return category
        
        if 'backbone' in module_name or 'PResNet' in module_name:
            return 'backbone'
        elif 'encoder' in module_name or 'HybridEncoder' in module_name:
            return 'encoder'
        elif 'decoder' in module_name or 'PolarFaceTransformer' in module_name:
            return 'decoder'
        elif 'criterion' in module_name or 'loss' in module_name:
            return 'loss_computation'
        elif 'dataloader' in module_name or 'dataset' in module_name:
            return 'data_loading'
        else:
            return 'other'
    
    def identify_bottlenecks(self):
        """
        Identify performance bottlenecks in the model.
        
        Returns:
            List of bottleneck descriptions
        """
        bottlenecks = []
        
        # Analyze tensor operations if not already done
        if not hasattr(self, 'tensor_ops') or self.tensor_ops is None:
            self.tensor_ops = self.analyze_tensor_operations()
        
        # Analyze modules if not already done
        if self.module_stats is None:
            self.analyze_modules()
        
        # Check for expensive tensor operations
        if not self.tensor_ops.empty:
            expensive_ops = self.tensor_ops[self.tensor_ops['percentage'] > 5].copy()
            for _, op in expensive_ops.iterrows():
                bottlenecks.append({
                    'type': 'tensor_operation',
                    'name': op['operation'],
                    'percentage': op['percentage'],
                    'calls': op['calls'],
                    'issues': op['issues'],
                    'recommendation': self._get_tensor_recommendation(op)
                })
        
        # Check for module bottlenecks
        if not self.module_stats.empty:
            expensive_modules = self.module_stats[self.module_stats['percentage'] > 3].copy()
            for _, module in expensive_modules.iterrows():
                if module['category'] in self.POLAR_BOTTLENECKS.keys():
                    bottlenecks.append({
                        'type': 'module',
                        'name': module['module'],
                        'category': module['category'],
                        'percentage': module['percentage'],
                        'calls': module['calls'],
                        'recommendation': self._get_module_recommendation(module)
                    })
        
        # Check for memory bottlenecks
        if self.memory_stats is not None and not self.memory_stats.empty:
            large_allocations = self.memory_stats[self.memory_stats['max_mb'] > 100].copy()
            for _, alloc in large_allocations.iterrows():
                bottlenecks.append({
                    'type': 'memory',
                    'name': alloc['operation'],
                    'total_mb': alloc['total_mb'],
                    'max_mb': alloc['max_mb'],
                    'allocations': alloc['allocations'],
                    'recommendation': self._get_memory_recommendation(alloc)
                })
        
        # Sort bottlenecks by importance
        self.bottlenecks = sorted(bottlenecks, key=lambda x: x.get('percentage', 0) 
                                  if 'percentage' in x else x.get('total_mb', 0), 
                                  reverse=True)
        
        return self.bottlenecks
    
    def _get_tensor_recommendation(self, op):
        """Generate recommendation for tensor operation bottleneck."""
        name = op['operation']
        issues = op['issues'] or []
        
        if 'Type conversion' in issues:
            return ("Minimize type conversions by ensuring consistent dtypes throughout the model. "
                   "Consider using mixed precision training with torch.cuda.amp.")
        
        if 'Memory-intensive concatenation' in issues:
            return ("Replace frequent concatenations with pre-allocated tensors or "
                   "more efficient operations. Consider using torch.cat only when necessary.")
        
        if 'Frequent reshaping' in issues:
            return ("Minimize reshape operations by maintaining consistent tensor shapes. "
                   "Consider refactoring to avoid repeated reshaping of the same tensors.")
        
        if 'Expensive math operation' in issues:
            if 'exp(' in name:
                return ("Expensive exponential operations detected. Consider using approximations "
                       "or optimizing Gaussian computations with vectorized operations.")
            
            return ("Expensive mathematical operations detected. Consider using more "
                   "efficient implementations or approximations.")
        
        if 'Consider mixed precision' in issues:
            return ("Enable automatic mixed precision (AMP) training to speed up float32 operations "
                   "by using float16 where appropriate.")
        
        return "Review implementation for potential optimizations."
    
    def _get_module_recommendation(self, module):
        """Generate recommendation for module bottleneck."""
        category = module['category']
        
        if category == 'gaussian_heatmap':
            return ("Optimize Gaussian heatmap generation with vectorized operations. "
                   "Consider pre-computing heatmaps or using approximations. "
                   "Disable heatmap generation during early training epochs.")
        
        elif category == 'polar_features':
            return ("Optimize polar feature extraction by vectorizing angle calculations. "
                   "Consider reducing polar bin resolution or using lookup tables.")
        
        elif category == 'transformer_attention':
            return ("Optimize attention mechanisms by reducing the number of attention heads "
                   "or using more efficient attention variants. Consider using FlashAttention "
                   "or other optimized attention implementations.")
        
        elif category == 'loss_computation':
            return ("Simplify loss computation by using fewer auxiliary losses during training. "
                   "Consider progressive loss activation during training.")
        
        elif category == 'data_loading':
            return ("Optimize data loading by increasing num_workers, enabling pin_memory, "
                   "and using persistent_workers. Pre-compute and cache expensive transformations.")
        
        return "Review implementation for potential optimizations."
    
    def _get_memory_recommendation(self, alloc):
        """Generate recommendation for memory bottleneck."""
        if alloc['allocations'] > 1000:
            return ("Frequent memory allocations detected. Use tensor pooling or "
                   "pre-allocation to reduce allocation overhead.")
        
        if alloc['max_mb'] > 500:
            return ("Large memory allocations detected. Consider reducing batch size, "
                   "using gradient checkpointing, or optimizing model architecture.")
        
        return ("Monitor memory usage and consider using smaller precision (e.g., float16) "
               "or gradient accumulation for large models.")
    
    def visualize_module_time(self, top_n=15, save_path=None):
        """
        Visualize time spent in each module.
        
        Args:
            top_n: Number of top modules to display
            save_path: Path to save the figure
        
        Returns:
            matplotlib figure
        """
        if self.module_stats is None:
            self.analyze_modules()
        
        if self.module_stats.empty:
            print("No module data available for visualization")
            return None
        
        # Get top N modules by total time
        top_modules = self.module_stats.head(top_n).copy()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        ax = sns.barplot(
            x='total_ms', 
            y='module', 
            hue='category',
            data=top_modules,
            palette='viridis'
        )
        
        # Add percentage labels
        for i, row in enumerate(top_modules.itertuples()):
            ax.text(
                row.total_ms + 0.5, 
                i, 
                f"{row.percentage:.1f}%", 
                va='center'
            )
        
        # Set labels and title
        plt.xlabel('Time (ms)')
        plt.ylabel('Module')
        plt.title('Time Spent in Top Modules')
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def visualize_memory_usage(self, top_n=10, save_path=None):
        """
        Visualize memory usage by operation.
        
        Args:
            top_n: Number of top operations to display
            save_path: Path to save the figure
        
        Returns:
            matplotlib figure
        """
        if self.memory_stats is None:
            self.analyze_memory_usage()
        
        if self.memory_stats.empty:
            print("No memory data available for visualization")
            return None
        
        # Get top N operations by total allocation
        top_ops = self.memory_stats.head(top_n).copy()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        ax = sns.barplot(
            x='total_mb', 
            y='operation', 
            data=top_ops,
            palette='coolwarm'
        )
        
        # Add allocation count labels
        for i, row in enumerate(top_ops.itertuples()):
            ax.text(
                row.total_mb + 0.5, 
                i, 
                f"{row.allocations} allocs", 
                va='center'
            )
        
        # Set labels and title
        plt.xlabel('Memory (MB)')
        plt.ylabel('Operation')
        plt.title('Memory Usage by Operation')
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_report(self, output_dir=None):
        """
        Generate a comprehensive performance report.
        
        Args:
            output_dir: Directory to save the report and figures
        
        Returns:
            Report as a string
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure all analyses are performed
        if self.module_stats is None:
            self.analyze_modules()
        
        tensor_ops = self.analyze_tensor_operations()
        
        if self.memory_stats is None:
            self.analyze_memory_usage()
        
        if not self.bottlenecks:
            self.identify_bottlenecks()
        
        # Generate report sections
        report = []
        report.append("# PolarRTDETRv2 Performance Profile Report")
        report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary section
        report.append("## Summary")
        if self.module_stats is not None and not self.module_stats.empty:
            total_time = self.module_stats['total_ms'].sum()
            report.append(f"- Total profiled time: {total_time:.2f} ms")
        
        if self.memory_stats is not None and not self.memory_stats.empty:
            total_memory = self.memory_stats['total_mb'].sum()
            report.append(f"- Total memory allocated: {total_memory:.2f} MB")
        
        report.append(f"- Number of identified bottlenecks: {len(self.bottlenecks)}")
        report.append("")
        
        # Bottlenecks section
        report.append("## Identified Bottlenecks")
        if self.bottlenecks:
            for i, bottleneck in enumerate(self.bottlenecks[:10], 1):
                report.append(f"### {i}. {bottleneck['name']}")
                report.append(f"- Type: {bottleneck['type']}")
                
                if 'category' in bottleneck:
                    report.append(f"- Category: {bottleneck['category']}")
                
                if 'percentage' in bottleneck:
                    report.append(f"- Time percentage: {bottleneck['percentage']:.2f}%")
                
                if 'total_mb' in bottleneck:
                    report.append(f"- Total memory: {bottleneck['total_mb']:.2f} MB")
                
                if 'calls' in bottleneck:
                    report.append(f"- Call count: {bottleneck['calls']}")
                
                if 'issues' in bottleneck and bottleneck['issues']:
                    report.append(f"- Issues: {', '.join(bottleneck['issues'])}")
                
                report.append(f"- Recommendation: {bottleneck['recommendation']}")
                report.append("")
        else:
            report.append("No significant bottlenecks identified.")
            report.append("")
        
        # Generate visualizations
        if output_dir:
            # Module time visualization
            fig = self.visualize_module_time(save_path=output_dir / "module_time.png")
            if fig:
                report.append("## Module Time Analysis")
                report.append("![Module Time](module_time.png)")
                report.append("")
            
            # Memory usage visualization
            fig = self.visualize_memory_usage(save_path=output_dir / "memory_usage.png")
            if fig:
                report.append("## Memory Usage Analysis")
                report.append("![Memory Usage](memory_usage.png)")
                report.append("")
        
        # Optimization recommendations section
        report.append("## Optimization Recommendations")
        report.append("### General Recommendations")
        report.append("1. **Enable Mixed Precision Training**: Use `torch.cuda.amp` to speed up computation.")
        report.append("2. **Optimize Batch Size**: Find the optimal batch size for your hardware.")
        report.append("3. **Data Loading**: Increase `num_workers` and enable `pin_memory` for faster data loading.")
        report.append("4. **Progressive Training**: Start with simpler losses and gradually add complexity.")
        report.append("")
        
        report.append("### Polar Face Specific Recommendations")
        report.append("1. **Gaussian Heatmap Generation**: Vectorize operations and consider disabling during early training.")
        report.append("2. **Landmark Processing**: Reduce landmark resolution during training.")
        report.append("3. **Polar Feature Extraction**: Optimize angle calculations with vectorized operations.")
        report.append("4. **Loss Functions**: Simplify loss computation by using fewer auxiliary losses.")
        report.append("")
        
        # Save report to file
        if output_dir:
            with open(output_dir / "profile_report.md", "w") as f:
                f.write("\n".join(report))
        
        return "\n".join(report)


def profile_model(model, input_shape=(1, 3, 640, 640), 
                 output_dir="./profile_output", 
                 warmup=2, active=5, use_cuda=True):
    """
    Profile a model with sample inputs.
    
    Args:
        model: PyTorch model to profile
        input_shape: Input tensor shape
        output_dir: Directory to save profiler output
        warmup: Number of warmup iterations
        active: Number of active profiling iterations
        use_cuda: Whether to use CUDA for profiling
    
    Returns:
        Path to the profile output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    print(f"Warming up for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
    
    # Profile
    print(f"Profiling for {active} iterations...")
    activities = [ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(active + warmup + 1):
                _ = model(sample_input)
                prof.step()
    
    # Save raw profile data
    prof.export_chrome_trace(str(output_dir / "trace.json"))
    
    print(f"Profile data saved to {output_dir}")
    return output_dir


def profile_training_step(model, criterion, sample_batch, 
                         output_dir="./profile_output", 
                         warmup=2, active=5, use_cuda=True):
    """
    Profile a complete training step including forward and backward pass.
    
    Args:
        model: PyTorch model to profile
        criterion: Loss function
        sample_batch: Sample batch of data (inputs, targets)
        output_dir: Directory to save profiler output
        warmup: Number of warmup iterations
        active: Number of active profiling iterations
        use_cuda: Whether to use CUDA for profiling
    
    Returns:
        Path to the profile output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    # Move sample batch to device
    inputs, targets = sample_batch
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)
    
    if isinstance(targets, list):
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in t.items()} for t in targets]
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Warmup
    print(f"Warming up for {warmup} iterations...")
    for _ in range(warmup):
        optimizer.zero_grad()
        outputs = model(inputs, targets=targets)
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
    
    # Profile
    print(f"Profiling for {active} iterations...")
    activities = [ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(active + warmup + 1):
            # Forward pass
            optimizer.zero_grad()
            with record_function("forward"):
                outputs = model(inputs, targets=targets)
            
            # Loss computation
            with record_function("loss_computation"):
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())
            
            # Backward pass
            with record_function("backward"):
                loss.backward()
            
            # Optimizer step
            with record_function("optimizer"):
                optimizer.step()
            
            prof.step()
    
    # Save raw profile data
    prof.export_chrome_trace(str(output_dir / "trace.json"))
    
    print(f"Profile data saved to {output_dir}")
    return output_dir


def analyze_polar_face_bottlenecks(profile_path, output_dir=None):
    """
    Analyze bottlenecks specific to PolarRTDETRv2 face detection model.
    
    Args:
        profile_path: Path to the profiler output directory
        output_dir: Directory to save the analysis report
    
    Returns:
        ProfileAnalyzer instance
    """
    analyzer = ProfileAnalyzer(profile_path)
    
    # Generate comprehensive report
    analyzer.generate_report(output_dir)
    
    # Print key bottlenecks
    bottlenecks = analyzer.identify_bottlenecks()
    if bottlenecks:
        print("\nTop 5 Performance Bottlenecks:")
        for i, bottleneck in enumerate(bottlenecks[:5], 1):
            print(f"{i}. {bottleneck['name']}")
            if 'percentage' in bottleneck:
                print(f"   - Time: {bottleneck['percentage']:.2f}%")
            elif 'total_mb' in bottleneck:
                print(f"   - Memory: {bottleneck['total_mb']:.2f} MB")
            print(f"   - Recommendation: {bottleneck['recommendation']}")
    
    return analyzer


def main():
    """Command line interface for profile analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PolarRTDETRv2 Profile Analyzer")
    parser.add_argument("profile_path", type=str, help="Path to the profiler output directory")
    parser.add_argument("--output", "-o", type=str, default="./profile_report",
                       help="Directory to save the analysis report")
    parser.add_argument("--top", "-t", type=int, default=15,
                       help="Number of top modules/operations to display")
    
    args = parser.parse_args()
    
    # Analyze profile data
    analyzer = analyze_polar_face_bottlenecks(args.profile_path, args.output)
    
    print(f"\nFull report saved to: {args.output}/profile_report.md")


if __name__ == "__main__":
    main()
