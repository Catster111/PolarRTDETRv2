#!/usr/bin/env python
"""
Annotation Verification Script for WiderFace Dataset with 5 Landmarks

This script verifies and validates the JSON annotation format for the WiderFace dataset
with 5 facial landmarks. It performs the following checks:
1. Validates the JSON structure
2. Checks for missing or invalid annotations
3. Visualizes sample images with annotations
4. Provides statistics about the dataset

Usage:
    python tools/verify_dataset.py --data-path /path/to/widerface --ann-file annotations/train_wider_face.json

Options:
    --data-path      Path to the WiderFace dataset directory
    --ann-file       Path to the annotation JSON file (relative to data-path)
    --image-prefix   Prefix to add to image paths from JSON (e.g., "WIDER_train/images/")
    --visualize      Enable visualization of samples (default: True)
    --num-samples    Number of random samples to visualize (default: 10)
    --output-dir     Directory to save visualization results (default: outputs/dataset_verification)
    --check-landmarks Check and validate landmarks (default: True)
    --verbose        Enable verbose output (default: True)
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import logging
from collections import defaultdict, Counter
import math

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('dataset_verification')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Verify WiderFace dataset with 5 landmarks')
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the WiderFace dataset directory')
    parser.add_argument('--ann-file', type=str, required=True,
                        help='Path to the annotation JSON file (relative to data-path)')
    parser.add_argument('--image-prefix', type=str, default='WIDER_train/images/',
                        help='Prefix to add to image paths from JSON (e.g., "WIDER_train/images/")')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Enable visualization of samples')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of random samples to visualize')
    parser.add_argument('--output-dir', type=str, default='outputs/dataset_verification',
                        help='Directory to save visualization results')
    parser.add_argument('--check-landmarks', action='store_true', default=True,
                        help='Check and validate landmarks')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Enable verbose output')
    
    return parser.parse_args()


def load_annotations(ann_file_path):
    """
    Load and parse the annotation JSON file.
    
    Args:
        ann_file_path: Path to the annotation JSON file
        
    Returns:
        annotations: Dictionary of annotations
    """
    try:
        with open(ann_file_path, 'r') as f:
            raw_ann = json.load(f)

        # ------------------------------------------------------------------
        # Convert COCO‐style JSON (with "images" & "annotations" arrays)
        # into the simpler mapping  {file_name: [ {bbox, landmarks, …}, … ]}
        # expected by the rest of this script.
        # ------------------------------------------------------------------
        if isinstance(raw_ann, dict) and "images" in raw_ann and "annotations" in raw_ann:
            id_to_file = {img["id"]: img["file_name"] for img in raw_ann["images"]}
            mapping: dict[str, list] = {fn: [] for fn in id_to_file.values()}

            for ann in raw_ann["annotations"]:
                file_name = id_to_file.get(ann["image_id"])
                if file_name is None:
                    continue

                # COCO bbox is [x, y, w, h]  →  convert to [x1, y1, x2, y2]
                x, y, w, h = ann["bbox"]
                bbox = [x, y, x + w, y + h]

                # COCO keypoints: [x, y, v] * 5  (15 values)
                kp = ann.get("keypoints", [])
                landmarks: list[float] = []
                if len(kp) == 15:
                    for j in range(0, 15, 3):
                        lx, ly, vis = kp[j], kp[j + 1], kp[j + 2]
                        landmarks.extend([0.0, 0.0] if vis == 0 else [lx, ly])
                if len(landmarks) != 10:
                    landmarks = [0.0] * 10

                mapping[file_name].append(
                    {
                        "bbox": bbox,
                        "landmarks": landmarks,
                        # Placeholder extra fields for consistency
                        "invalid": ann.get("iscrowd", 0),
                    }
                )

            # remove images without faces
            annotations = {k: v for k, v in mapping.items() if v}
        else:
            # Assume already in mapping format
            annotations = raw_ann
        
        logger.info(f"Successfully loaded annotations from {ann_file_path}")
        return annotations
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {ann_file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading annotations from {ann_file_path}: {e}")
        sys.exit(1)


def validate_annotation_structure(annotations):
    """
    Validate the structure of the annotation dictionary.
    
    Args:
        annotations: Dictionary of annotations
        
    Returns:
        is_valid: Whether the critical structure is valid (bbox present & correct)
        errors:   List of *blocking* errors (script should stop)
        warnings: List of *non-blocking* warnings (script can continue)
    """
    errors: list[str] = []
    warnings: list[str] = []
    
    if not isinstance(annotations, dict):
        errors.append("Annotations should be a dictionary mapping image paths to lists of faces")
        return False, errors, warnings
    
    for img_path, faces in annotations.items():
        # Check image path
        if not isinstance(img_path, str):
            errors.append(f"Image path should be a string, got {type(img_path)}")
            continue   # skip further checks for this entry
        
        # Check faces list
        if not isinstance(faces, list):
            errors.append(f"Faces for image {img_path} should be a list, got {type(faces)}")
            continue
        
        # Check each face annotation
        for i, face in enumerate(faces):
            if not isinstance(face, dict):
                errors.append(
                    f"Face annotation for image {img_path}, face #{i} should be a dictionary, got {type(face)}"
                )
                continue
            
            # Check bbox
            if "bbox" not in face:
                errors.append(f"Missing 'bbox' in face annotation for image {img_path}, face #{i}")
            elif not isinstance(face["bbox"], list) or len(face["bbox"]) != 4:
                errors.append(
                    f"'bbox' should be a list of 4 values, got {face['bbox']} for image {img_path}, face #{i}"
                )
            
            # Check landmarks
            if "landmarks" not in face:
                warnings.append(
                    f"Missing 'landmarks' in face annotation for image {img_path}, face #{i}"
                )
            elif not isinstance(face["landmarks"], list) or len(face["landmarks"]) != 10:
                warnings.append(
                    f"'landmarks' should be a list of 10 values (5 landmarks with x,y coordinates), "
                    f"got {len(face['landmarks'])} values for image {img_path}, face #{i}"
                )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def check_image_existence(data_path, annotations, image_prefix=''):
    """
    Check if all images in annotations exist on disk.
    
    Args:
        data_path: Path to the dataset directory
        annotations: Dictionary of annotations
        image_prefix: Prefix to add to image paths from JSON
        
    Returns:
        missing_images: List of missing image paths
    """
    missing_images = []
    
    for img_path in tqdm(annotations.keys(), desc="Checking image existence"):
        prefixed_path = os.path.join(image_prefix, img_path) if image_prefix else img_path
        full_path = os.path.join(data_path, prefixed_path)
        if not os.path.exists(full_path):
            missing_images.append(img_path)
    
    return missing_images


def validate_bboxes(annotations, data_path, image_prefix=''):
    """
    Validate bounding boxes in annotations.
    
    Args:
        annotations: Dictionary of annotations
        data_path: Path to the dataset directory
        image_prefix: Prefix to add to image paths from JSON
        
    Returns:
        invalid_bboxes: Dictionary mapping image paths to lists of invalid bbox indices
        bbox_stats: Dictionary of bounding box statistics
    """
    invalid_bboxes = {}
    bbox_stats = {
        'total': 0,
        'invalid_coords': 0,
        'outside_image': 0,
        'zero_area': 0,
        'negative_size': 0,
        'width_distribution': defaultdict(int),
        'height_distribution': defaultdict(int),
        'aspect_ratio_distribution': defaultdict(int)
    }
    
    for img_path, faces in tqdm(annotations.items(), desc="Validating bounding boxes"):
        # Get image dimensions
        try:
            prefixed_path = os.path.join(image_prefix, img_path) if image_prefix else img_path
            full_path = os.path.join(data_path, prefixed_path)
            img = cv2.imread(full_path)
            if img is None:
                logger.warning(f"Could not read image {full_path}")
                continue
                
            img_height, img_width = img.shape[:2]
        except Exception as e:
            logger.warning(f"Error reading image {img_path}: {e}")
            continue
        
        invalid_in_image = []
        
        for i, face in enumerate(faces):
            bbox_stats['total'] += 1
            
            # Get bbox coordinates
            bbox = face['bbox']
            
            # Check if bbox is in the correct format
            if len(bbox) != 4:
                bbox_stats['invalid_coords'] += 1
                invalid_in_image.append(i)
                continue
            
            # Check if bbox is in [x, y, w, h] format and convert to [x1, y1, x2, y2]
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                # This is likely [x, y, w, h] format
                x, y, w, h = bbox
                bbox = [x, y, x + w, y + h]
            
            x1, y1, x2, y2 = bbox
            
            # Check for negative size
            if x2 <= x1 or y2 <= y1:
                bbox_stats['negative_size'] += 1
                invalid_in_image.append(i)
                continue
            
            # Check for zero area
            if (x2 - x1) * (y2 - y1) <= 0:
                bbox_stats['zero_area'] += 1
                invalid_in_image.append(i)
                continue
            
            # Check if bbox is outside image
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                bbox_stats['outside_image'] += 1
                invalid_in_image.append(i)
                continue
            
            # Calculate statistics
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            # Bin width and height
            width_bin = int(width / 10) * 10  # Bin to nearest 10 pixels
            height_bin = int(height / 10) * 10  # Bin to nearest 10 pixels
            aspect_ratio_bin = round(aspect_ratio * 2) / 2  # Bin to nearest 0.5
            
            bbox_stats['width_distribution'][width_bin] += 1
            bbox_stats['height_distribution'][height_bin] += 1
            bbox_stats['aspect_ratio_distribution'][aspect_ratio_bin] += 1
        
        if invalid_in_image:
            invalid_bboxes[img_path] = invalid_in_image
    
    return invalid_bboxes, bbox_stats


def validate_landmarks(annotations, check_visibility=True):
    """
    Validate landmarks in annotations.
    
    Args:
        annotations: Dictionary of annotations
        check_visibility: Whether to check landmark visibility
        
    Returns:
        invalid_landmarks: Dictionary mapping image paths to lists of invalid landmark indices
        landmark_stats: Dictionary of landmark statistics
    """
    invalid_landmarks = {}
    landmark_stats = {
        'total': 0,
        'missing': 0,
        'invalid_coords': 0,
        'outside_bbox': 0,
        'all_zero': 0,
        'visibility': defaultdict(int)
    }
    
    for img_path, faces in tqdm(annotations.items(), desc="Validating landmarks"):
        invalid_in_image = []
        
        for i, face in enumerate(faces):
            landmark_stats['total'] += 1
            
            # Check if landmarks exist
            if 'landmarks' not in face:
                landmark_stats['missing'] += 1
                invalid_in_image.append(i)
                continue
            
            landmarks = face['landmarks']
            
            # Check if landmarks have the correct length
            if len(landmarks) != 10:  # 5 landmarks, each with x and y
                landmark_stats['invalid_coords'] += 1
                invalid_in_image.append(i)
                continue
            
            # Check if all landmarks are zero (invalid)
            if all(lm == 0 for lm in landmarks):
                landmark_stats['all_zero'] += 1
                invalid_in_image.append(i)
                continue
            
            # Check if landmarks are inside bbox
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox if len(bbox) == 4 else [0, 0, 0, 0]
            
            landmarks_outside_bbox = False
            for j in range(0, len(landmarks), 2):
                lm_x, lm_y = landmarks[j], landmarks[j+1]
                
                # Skip if landmark is not visible (zero coordinates)
                if lm_x == 0 and lm_y == 0:
                    continue
                
                # Check if landmark is outside bbox
                if lm_x < x1 or lm_x > x2 or lm_y < y1 or lm_y > y2:
                    landmarks_outside_bbox = True
                    break
            
            if landmarks_outside_bbox:
                landmark_stats['outside_bbox'] += 1
                invalid_in_image.append(i)
            
            # Count visibility of each landmark
            if check_visibility:
                for j in range(0, len(landmarks), 2):
                    lm_x, lm_y = landmarks[j], landmarks[j+1]
                    lm_idx = j // 2  # 0 to 4
                    
                    # Landmark is visible if both coordinates are non-zero
                    is_visible = not (lm_x == 0 and lm_y == 0)
                    landmark_stats['visibility'][lm_idx] = landmark_stats['visibility'].get(lm_idx, 0) + int(is_visible)
        
        if invalid_in_image:
            invalid_landmarks[img_path] = invalid_in_image
    
    return invalid_landmarks, landmark_stats


def visualize_samples(annotations, data_path, num_samples, output_dir, image_prefix=''):
    """
    Visualize random samples from the dataset.
    
    Args:
        annotations: Dictionary of annotations
        data_path: Path to the dataset directory
        num_samples: Number of random samples to visualize
        output_dir: Directory to save visualization results
        image_prefix: Prefix to add to image paths from JSON
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images with annotations
    image_paths = list(annotations.keys())
    
    # Select random samples
    if len(image_paths) <= num_samples:
        samples = image_paths
    else:
        samples = random.sample(image_paths, num_samples)
    
    # Define colors for visualization
    bbox_color = (0, 255, 0)  # Green
    landmark_colors = [
        (255, 0, 0),    # Red - right eye
        (0, 0, 255),    # Blue - left eye
        (255, 255, 0),  # Yellow - nose
        (255, 0, 255),  # Magenta - right mouth corner
        (0, 255, 255)   # Cyan - left mouth corner
    ]
    
    for i, img_path in enumerate(samples):
        # Load image
        prefixed_path = os.path.join(image_prefix, img_path) if image_prefix else img_path
        full_path = os.path.join(data_path, prefixed_path)
        img = cv2.imread(full_path)
        
        if img is None:
            logger.warning(f"Could not read image {full_path}")
            continue
        
        # Get faces for this image
        faces = annotations[img_path]
        
        # Draw bounding boxes and landmarks
        for face in faces:
            # Draw bbox
            bbox = face['bbox']
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 2)
            
            # Draw landmarks
            if 'landmarks' in face:
                landmarks = face['landmarks']
                for j in range(0, len(landmarks), 2):
                    lm_idx = j // 2
                    lm_x, lm_y = landmarks[j], landmarks[j+1]
                    
                    # Skip if landmark is not visible
                    if lm_x == 0 and lm_y == 0:
                        continue
                    
                    # Draw landmark
                    cv2.circle(img, (int(lm_x), int(lm_y)), 3, landmark_colors[lm_idx], -1)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"sample_{i+1}.jpg")
        cv2.imwrite(output_path, img)
        
        # Create a figure with a larger size for better visibility
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Sample {i+1}: {img_path}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{i+1}_plot.png"), dpi=150)
        plt.close()
    
    logger.info(f"Saved {len(samples)} visualizations to {output_dir}")


def plot_statistics(bbox_stats, landmark_stats, output_dir):
    """
    Plot statistics about the dataset.
    
    Args:
        bbox_stats: Dictionary of bounding box statistics
        landmark_stats: Dictionary of landmark statistics
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot bbox width distribution
    plt.figure(figsize=(12, 6))
    widths = sorted(bbox_stats['width_distribution'].keys())
    width_counts = [bbox_stats['width_distribution'][w] for w in widths]
    plt.bar(range(len(widths)), width_counts)
    plt.xticks(range(len(widths)), widths, rotation=90)
    plt.xlabel('Width (pixels)')
    plt.ylabel('Count')
    plt.title('Bounding Box Width Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbox_width_distribution.png'))
    plt.close()
    
    # Plot bbox height distribution
    plt.figure(figsize=(12, 6))
    heights = sorted(bbox_stats['height_distribution'].keys())
    height_counts = [bbox_stats['height_distribution'][h] for h in heights]
    plt.bar(range(len(heights)), height_counts)
    plt.xticks(range(len(heights)), heights, rotation=90)
    plt.xlabel('Height (pixels)')
    plt.ylabel('Count')
    plt.title('Bounding Box Height Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbox_height_distribution.png'))
    plt.close()
    
    # Plot bbox aspect ratio distribution
    plt.figure(figsize=(12, 6))
    aspect_ratios = sorted(bbox_stats['aspect_ratio_distribution'].keys())
    aspect_ratio_counts = [bbox_stats['aspect_ratio_distribution'][ar] for ar in aspect_ratios]
    plt.bar(range(len(aspect_ratios)), aspect_ratio_counts)
    plt.xticks(range(len(aspect_ratios)), [f"{ar:.1f}" for ar in aspect_ratios], rotation=90)
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Count')
    plt.title('Bounding Box Aspect Ratio Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbox_aspect_ratio_distribution.png'))
    plt.close()
    
    # Plot landmark visibility
    if landmark_stats['visibility']:
        plt.figure(figsize=(10, 6))
        landmark_names = ['Right Eye', 'Left Eye', 'Nose', 'Right Mouth', 'Left Mouth']
        landmark_indices = sorted(landmark_stats['visibility'].keys())
        visibility_counts = [landmark_stats['visibility'][idx] for idx in landmark_indices]
        visibility_percentage = [count / landmark_stats['total'] * 100 for count in visibility_counts]
        
        plt.bar(landmark_names, visibility_percentage)
        plt.ylabel('Visibility (%)')
        plt.title('Landmark Visibility')
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'landmark_visibility.png'))
        plt.close()
    
    # Create a summary of statistics
    with open(os.path.join(output_dir, 'statistics_summary.txt'), 'w') as f:
        f.write("=== Dataset Statistics Summary ===\n\n")
        
        f.write("--- Bounding Box Statistics ---\n")
        f.write(f"Total bounding boxes: {bbox_stats['total']}\n")
        f.write(f"Invalid coordinates: {bbox_stats['invalid_coords']} ({bbox_stats['invalid_coords'] / max(1, bbox_stats['total']) * 100:.2f}%)\n")
        f.write(f"Outside image: {bbox_stats['outside_image']} ({bbox_stats['outside_image'] / max(1, bbox_stats['total']) * 100:.2f}%)\n")
        f.write(f"Zero area: {bbox_stats['zero_area']} ({bbox_stats['zero_area'] / max(1, bbox_stats['total']) * 100:.2f}%)\n")
        f.write(f"Negative size: {bbox_stats['negative_size']} ({bbox_stats['negative_size'] / max(1, bbox_stats['total']) * 100:.2f}%)\n\n")
        
        f.write("--- Landmark Statistics ---\n")
        f.write(f"Total faces with landmarks: {landmark_stats['total']}\n")
        f.write(f"Missing landmarks: {landmark_stats['missing']} ({landmark_stats['missing'] / max(1, landmark_stats['total']) * 100:.2f}%)\n")
        f.write(f"Invalid coordinates: {landmark_stats['invalid_coords']} ({landmark_stats['invalid_coords'] / max(1, landmark_stats['total']) * 100:.2f}%)\n")
        f.write(f"Outside bbox: {landmark_stats['outside_bbox']} ({landmark_stats['outside_bbox'] / max(1, landmark_stats['total']) * 100:.2f}%)\n")
        f.write(f"All zero (invisible): {landmark_stats['all_zero']} ({landmark_stats['all_zero'] / max(1, landmark_stats['total']) * 100:.2f}%)\n\n")
        
        if landmark_stats['visibility']:
            f.write("--- Landmark Visibility ---\n")
            landmark_names = ['Right Eye', 'Left Eye', 'Nose', 'Right Mouth', 'Left Mouth']
            for idx, name in enumerate(landmark_names):
                visibility = landmark_stats['visibility'].get(idx, 0)
                percentage = visibility / landmark_stats['total'] * 100 if landmark_stats['total'] > 0 else 0
                f.write(f"{name}: {visibility} ({percentage:.2f}%)\n")
    
    logger.info(f"Saved statistics plots and summary to {output_dir}")


def print_dataset_summary(annotations, invalid_bboxes, invalid_landmarks, bbox_stats, landmark_stats):
    """
    Print a summary of the dataset.
    
    Args:
        annotations: Dictionary of annotations
        invalid_bboxes: Dictionary of invalid bounding boxes
        invalid_landmarks: Dictionary of invalid landmarks
        bbox_stats: Dictionary of bounding box statistics
        landmark_stats: Dictionary of landmark statistics
    """
    total_images = len(annotations)
    total_faces = sum(len(faces) for faces in annotations.values())
    images_with_invalid_bboxes = len(invalid_bboxes)
    images_with_invalid_landmarks = len(invalid_landmarks)
    
    print("\n=== DATASET SUMMARY ===")
    print(f"Total images: {total_images}")
    print(f"Total faces: {total_faces}")
    print(f"Average faces per image: {total_faces / total_images:.2f}")
    print(f"Images with invalid bboxes: {images_with_invalid_bboxes} ({images_with_invalid_bboxes / total_images * 100:.2f}%)")
    print(f"Images with invalid landmarks: {images_with_invalid_landmarks} ({images_with_invalid_landmarks / total_images * 100:.2f}%)")
    
    print("\n--- Bounding Box Statistics ---")
    print(f"Total bounding boxes: {bbox_stats['total']}")
    total_bbox = max(1, bbox_stats['total'])
    print(f"Invalid coordinates: {bbox_stats['invalid_coords']} ({bbox_stats['invalid_coords'] / total_bbox * 100:.2f}%)")
    print(f"Outside image: {bbox_stats['outside_image']} ({bbox_stats['outside_image'] / total_bbox * 100:.2f}%)")
    print(f"Zero area: {bbox_stats['zero_area']} ({bbox_stats['zero_area'] / total_bbox * 100:.2f}%)")
    print(f"Negative size: {bbox_stats['negative_size']} ({bbox_stats['negative_size'] / total_bbox * 100:.2f}%)")
    
    # Calculate average and median sizes
    widths = []
    heights = []
    for width, count in bbox_stats['width_distribution'].items():
        widths.extend([width] * count)
    for height, count in bbox_stats['height_distribution'].items():
        heights.extend([height] * count)
    
    if widths and heights:
        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)
        median_width = sorted(widths)[len(widths) // 2]
        median_height = sorted(heights)[len(heights) // 2]
        
        print(f"Average bbox size: {avg_width:.1f} x {avg_height:.1f} pixels")
        print(f"Median bbox size: {median_width} x {median_height} pixels")
    
    print("\n--- Landmark Statistics ---")
    print(f"Total faces with landmarks: {landmark_stats['total']}")
    print(f"Missing landmarks: {landmark_stats['missing']} ({landmark_stats['missing'] / landmark_stats['total'] * 100:.2f}%)")
    print(f"Invalid coordinates: {landmark_stats['invalid_coords']} ({landmark_stats['invalid_coords'] / landmark_stats['total'] * 100:.2f}%)")
    print(f"Outside bbox: {landmark_stats['outside_bbox']} ({landmark_stats['outside_bbox'] / landmark_stats['total'] * 100:.2f}%)")
    print(f"All zero (invisible): {landmark_stats['all_zero']} ({landmark_stats['all_zero'] / landmark_stats['total'] * 100:.2f}%)")
    
    if landmark_stats['visibility']:
        print("\n--- Landmark Visibility ---")
        landmark_names = ['Right Eye', 'Left Eye', 'Nose', 'Right Mouth', 'Left Mouth']
        for idx, name in enumerate(landmark_names):
            visibility = landmark_stats['visibility'].get(idx, 0)
            percentage = visibility / landmark_stats['total'] * 100
            print(f"{name}: {visibility} ({percentage:.2f}%)")
    
    print("\n=========================")


def main():
    """Main function."""
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(output_dir / 'verification.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting dataset verification")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Annotation file: {args.ann_file}")
    
    # Load annotations
    ann_file_path = os.path.join(args.data_path, args.ann_file)
    annotations = load_annotations(ann_file_path)
    
    # Validate annotation structure
    logger.info("Validating annotation structure")
    is_valid, errors, warnings = validate_annotation_structure(annotations)
    
    if not is_valid:
        logger.error("Invalid annotation structure:")
        for err in errors:
            logger.error(f"  - {err}")
        
        if args.verbose:
            print("Invalid annotation structure. See log for details.")
        
        sys.exit(1)
    else:
        # Log any non-blocking warnings
        if warnings:
            logger.warning(f"Found {len(warnings)} warnings in annotation structure")
            for w in warnings[:20]:  # limit console spam
                logger.warning(f"  - {w}")
            if args.verbose:
                print(f"Warning: {len(warnings)} non-blocking issues found in annotations. "
                      f"See log for full list.")
    
    logger.info("Annotation structure is valid")
    
    # Check image existence
    logger.info("Checking image existence")
    missing_images = check_image_existence(args.data_path, annotations, args.image_prefix)
    
    if missing_images:
        logger.warning(f"Found {len(missing_images)} missing images")
        if args.verbose:
            print(f"Warning: Found {len(missing_images)} missing images")
            if len(missing_images) <= 10:
                for img_path in missing_images:
                    print(f"  - {img_path}")
            else:
                for img_path in missing_images[:10]:
                    print(f"  - {img_path}")
                print(f"  ... and {len(missing_images) - 10} more")
    else:
        logger.info("All images exist")
    
    # Validate bounding boxes
    logger.info("Validating bounding boxes")
    invalid_bboxes, bbox_stats = validate_bboxes(annotations, args.data_path, args.image_prefix)
    
    if invalid_bboxes:
        logger.warning(f"Found {len(invalid_bboxes)} images with invalid bounding boxes")
        if args.verbose:
            print(f"Warning: Found {len(invalid_bboxes)} images with invalid bounding boxes")
    else:
        logger.info("All bounding boxes are valid")
    
    # Validate landmarks
    if args.check_landmarks:
        logger.info("Validating landmarks")
        invalid_landmarks, landmark_stats = validate_landmarks(annotations)
        
        if invalid_landmarks:
            logger.warning(f"Found {len(invalid_landmarks)} images with invalid landmarks")
            if args.verbose:
                print(f"Warning: Found {len(invalid_landmarks)} images with invalid landmarks")
        else:
            logger.info("All landmarks are valid")
    else:
        invalid_landmarks = {}
        landmark_stats = {'total': 0}
    
    # Print dataset summary
    if args.verbose:
        print_dataset_summary(annotations, invalid_bboxes, invalid_landmarks, bbox_stats, landmark_stats)
    
    # Visualize samples
    if args.visualize:
        logger.info(f"Visualizing {args.num_samples} random samples")
        visualize_samples(annotations, args.data_path, args.num_samples, 
                         os.path.join(output_dir, 'visualizations'), args.image_prefix)
    
    # Plot statistics
    logger.info("Plotting statistics")
    plot_statistics(bbox_stats, landmark_stats, os.path.join(output_dir, 'statistics'))
    
    logger.info("Dataset verification completed")
    
    if args.verbose:
        print(f"\nVerification completed. Results saved to {output_dir}")
        print(f"- Log: {output_dir}/verification.log")
        print(f"- Visualizations: {output_dir}/visualizations/")
        print(f"- Statistics: {output_dir}/statistics/")


if __name__ == '__main__':
    main()
