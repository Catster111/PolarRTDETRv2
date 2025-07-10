#!/usr/bin/env python
"""
Test script for Polar-RTDETRv2 dataset loading.

This script tests the dataset loading functionality for the Polar-RTDETRv2 project.
It loads the WiderFace dataset with 5 landmarks and displays sample images and annotations.

Usage:
    python run_test.py --data-path /path/to/widerface --ann-file annotations/train_wider_face.json --image-prefix WIDER_train/images/
"""

import os
import sys
import argparse
import random
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import polar_datasets module
from polar_datasets import build_widerface
from polar_datasets.transforms import ToTensor, Normalize, Compose


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('dataset_test')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test WiderFace dataset loading')
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the WiderFace dataset directory')
    parser.add_argument('--ann-file', type=str, required=True,
                        help='Path to the annotation JSON file (relative to data-path)')
    parser.add_argument('--image-prefix', type=str, default='WIDER_train/images/',
                        help='Prefix to add to image paths from JSON (e.g., "WIDER_train/images/")')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of random samples to visualize')
    parser.add_argument('--use-polar', action='store_true',
                        help='Use polar coordinate representation')
    parser.add_argument('--output-dir', type=str, default='test_outputs',
                        help='Directory to save visualization results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def visualize_sample(img, target, idx, output_dir, use_polar=False):
    """
    Visualize a sample image with annotations.
    
    Args:
        img: PIL Image or tensor
        target: Target dictionary with annotations
        idx: Sample index
        output_dir: Output directory for saving visualizations
        use_polar: Whether polar coordinates are used
    """
    # Convert tensor to numpy if needed
    if isinstance(img, torch.Tensor):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = np.array(img)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(img_np)
    
    # Get image dimensions
    height, width = img_np.shape[:2]
    # pre-compute centre and diagonal for polar conversions
    center_x, center_y = width / 2.0, height / 2.0
    diag = np.sqrt(width ** 2 + height ** 2)
    
    # Define colors for visualization
    bbox_color = 'g'  # Green for bounding boxes
    landmark_colors = ['r', 'b', 'y', 'm', 'c']  # Colors for each landmark
    
    # Draw bounding boxes
    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes'].cpu().numpy()
        
        # Helpers to convert polar → Cartesian ------------------------
        def _polar_pt_to_xy(r_val, theta_val):
            """
            Convert one polar point to (x,y) in image coords.
            Radius was stored normalised by the image diagonal during dataset
            creation, so we need to denormalise first.
            """
            diag = np.sqrt(width ** 2 + height ** 2)
            r_denorm = r_val * diag
            x_pt = center_x + r_denorm * np.cos(theta_val)
            y_pt = center_y + r_denorm * np.sin(theta_val)
            return x_pt, y_pt

        def _polar_box_to_xyxy(p_box):
            """
            Convert polar bbox [r1,theta1,r2,theta2] back to [x1,y1,x2,y2].
            """
            x1, y1 = _polar_pt_to_xy(p_box[0], p_box[1])
            x2, y2 = _polar_pt_to_xy(p_box[2], p_box[3])
            # Ensure ordering (left,top,right,bottom)
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            return [x_min, y_min, x_max, y_max]

        # Convert polar -> Cartesian for boxes / landmarks if requested
        if use_polar and target.get('use_polar', False):
            # Convert each bbox
            boxes_cart = np.array([_polar_box_to_xyxy(b) for b in boxes])
            boxes = boxes_cart
        
        # Draw each box
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=bbox_color, linewidth=2)
            )
            
            # Draw landmarks if available
            if 'landmarks' in target and len(target['landmarks']) > 0:
                landmarks = target['landmarks'][box_idx].cpu().numpy().copy()
                # Convert polar to Cartesian if needed
                if use_polar and target.get('use_polar', False):
                    for lm_i in range(0, len(landmarks), 2):
                        r_val, theta_val = landmarks[lm_i], landmarks[lm_i + 1]
                        if r_val == 0 and theta_val == 0:  # invisible
                            continue
                        # de-normalize radius then convert
                        r_denorm = r_val * diag
                        x_cart = center_x + r_denorm * np.cos(theta_val)
                        y_cart = center_y + r_denorm * np.sin(theta_val)
                        landmarks[lm_i] = x_cart
                        landmarks[lm_i + 1] = y_cart
                
                # Draw each landmark
                for lm_idx in range(5):  # 5 landmarks
                    x, y = landmarks[lm_idx*2], landmarks[lm_idx*2 + 1]
                    
                    # Skip if landmark is not visible (zero coordinates)
                    if x == 0 and y == 0:
                        continue
                    
                    plt.plot(x, y, 'o', color=landmark_colors[lm_idx], markersize=8)
    
    # Add title and labels
    plt.title(f"Sample {idx}")
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"sample_{idx}.png"), dpi=150)
    plt.close()


def test_dataset_loading(args):
    """
    Test dataset loading functionality.
    
    Args:
        args: Command line arguments
    """
    logger.info("Testing dataset loading...")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Annotation file: {args.ann_file}")
    logger.info(f"Image prefix: {args.image_prefix}")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create a simple class to hold dataset arguments
    class DatasetArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Create dataset arguments
    dataset_args = DatasetArgs(
        widerface_path=args.data_path,
        use_polar=args.use_polar,
        filter_invalid=True,
        min_face_size=8,
        cache_mode=False,
        image_prefix=args.image_prefix
    )
    
    # Create simple transforms
    transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        dataset, num_classes = build_widerface('train', dataset_args)
        
        # Print dataset information
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Number of images: {len(dataset)}")
        logger.info(f"Number of classes: {num_classes}")
        
        # Get some sample indices
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
        
        # Visualize samples
        logger.info(f"Visualizing {len(indices)} random samples...")
        for i, idx in enumerate(indices):
            try:
                # Get sample
                img, target = dataset[idx]
                
                # Print sample information
                logger.info(f"Sample {i+1}/{len(indices)} (index {idx}):")
                logger.info(f"  Image path: {target.get('img_path', 'N/A')}")
                logger.info(f"  Image size: {tuple(target['orig_size'].tolist())}")
                logger.info(f"  Number of faces: {len(target['boxes'])}")
                
                if 'landmarks' in target:
                    visible_landmarks = torch.sum(target['landmarks'] != 0).item() // 2
                    total_landmarks = len(target['landmarks']) * 5
                    logger.info(f"  Visible landmarks: {visible_landmarks}/{total_landmarks}")
                
                # Visualize sample
                visualize_sample(img, target, i+1, args.output_dir, args.use_polar)
                logger.info(f"  Visualization saved to {os.path.join(args.output_dir, f'sample_{i+1}.png')}")
            
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
        
        logger.info("Dataset test completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    args = parse_args()
    success = test_dataset_loading(args)
    
    if success:
        logger.info("Dataset loading test passed!")
        sys.exit(0)
    else:
        logger.error("Dataset loading test failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
