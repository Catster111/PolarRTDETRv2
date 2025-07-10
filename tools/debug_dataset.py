#!/usr/bin/env python
"""
Dataset Debug Script for WiderFace with 5 Landmarks

This script helps diagnose and fix issues with the WiderFace dataset structure,
particularly focusing on path mismatches between JSON annotations and actual files.
It provides detailed information about:

1. JSON structure and content
2. Actual directory structure on disk
3. Path mapping issues and suggested fixes
4. Detailed statistics about bounding boxes and landmarks

Usage:
    python tools/debug_dataset.py --data-path /path/to/dataset --ann-file annotations/train_wider_face.json

Options:
    --data-path      Path to the dataset root directory
    --ann-file       Path to the annotation JSON file (relative to data-path)
    --fix-paths      Generate a fixed JSON file with corrected paths
    --output-file    Path to save the fixed JSON file (default: fixed_annotations.json)
    --check-first    Number of images to check first (default: 20, use -1 for all)
    --verbose        Enable verbose output
"""

import os
import sys
import json
import argparse
from pathlib import Path
import logging
from collections import defaultdict, Counter
import glob
import re
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('dataset_debug')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Debug WiderFace dataset structure')
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the dataset root directory')
    parser.add_argument('--ann-file', type=str, required=True,
                        help='Path to the annotation JSON file (relative to data-path)')
    parser.add_argument('--fix-paths', action='store_true',
                        help='Generate a fixed JSON file with corrected paths')
    parser.add_argument('--output-file', type=str, default='fixed_annotations.json',
                        help='Path to save the fixed JSON file')
    parser.add_argument('--check-first', type=int, default=20,
                        help='Number of images to check first (use -1 for all)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def load_json(file_path):
    """Load JSON file and return its content."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        sys.exit(1)


def analyze_json_structure(json_data):
    """
    Analyze the structure of the JSON data.
    
    Args:
        json_data: Loaded JSON data
        
    Returns:
        structure_info: Dictionary with structure information
    """
    structure_info = {
        'format': 'unknown',
        'keys': list(json_data.keys()) if isinstance(json_data, dict) else [],
        'is_coco': False,
        'image_count': 0,
        'annotation_count': 0,
        'category_count': 0,
        'sample_image_paths': [],
        'sample_bbox_format': None,
        'sample_keypoint_format': None
    }
    
    # Check if it's a COCO-format JSON
    if isinstance(json_data, dict) and all(k in json_data for k in ['images', 'annotations']):
        structure_info['format'] = 'coco'
        structure_info['is_coco'] = True
        structure_info['image_count'] = len(json_data.get('images', []))
        structure_info['annotation_count'] = len(json_data.get('annotations', []))
        structure_info['category_count'] = len(json_data.get('categories', []))
        
        # Get sample image paths
        for img in json_data.get('images', [])[:5]:
            structure_info['sample_image_paths'].append(img.get('file_name', ''))
        
        # Get sample bbox and keypoint format
        if json_data.get('annotations'):
            ann = json_data['annotations'][0]
            structure_info['sample_bbox_format'] = ann.get('bbox', [])
            structure_info['sample_keypoint_format'] = ann.get('keypoints', [])
    
    # Check if it's a simple mapping format
    elif isinstance(json_data, dict) and all(isinstance(v, list) for v in json_data.values()):
        structure_info['format'] = 'simple_mapping'
        structure_info['image_count'] = len(json_data)
        structure_info['annotation_count'] = sum(len(faces) for faces in json_data.values())
        
        # Get sample image paths
        structure_info['sample_image_paths'] = list(json_data.keys())[:5]
        
        # Get sample bbox and keypoint format
        for img_path, faces in json_data.items():
            if faces:
                structure_info['sample_bbox_format'] = faces[0].get('bbox', [])
                structure_info['sample_keypoint_format'] = faces[0].get('landmarks', [])
                break
    
    return structure_info


def analyze_directory_structure(data_path, check_subdirs=True):
    """
    Analyze the directory structure of the dataset.
    
    Args:
        data_path: Path to the dataset root directory
        check_subdirs: Whether to check subdirectories
        
    Returns:
        dir_info: Dictionary with directory information
    """
    dir_info = {
        'exists': os.path.exists(data_path),
        'is_dir': os.path.isdir(data_path),
        'subdirs': [],
        'image_dirs': [],
        'image_count': 0,
        'sample_image_paths': []
    }
    
    if not dir_info['exists'] or not dir_info['is_dir']:
        return dir_info
    
    # Get subdirectories
    subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    dir_info['subdirs'] = subdirs
    
    # Check for common WiderFace directories
    widerface_dirs = [
        'WIDER_train', 'WIDER_val', 'WIDER_test',
        'train', 'val', 'test',
        'images', 'annotations'
    ]
    
    for d in widerface_dirs:
        if d in subdirs:
            subdir_path = os.path.join(data_path, d)
            
            # Check if this directory contains images
            image_count = len(glob.glob(os.path.join(subdir_path, '**/*.jpg'), recursive=True))
            if image_count > 0:
                dir_info['image_dirs'].append({
                    'path': d,
                    'image_count': image_count,
                    'has_subdirs': len([sd for sd in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, sd))]) > 0
                })
                dir_info['image_count'] += image_count
    
    # Get sample image paths
    all_images = []
    for img_dir in dir_info['image_dirs']:
        dir_path = os.path.join(data_path, img_dir['path'])
        images = glob.glob(os.path.join(dir_path, '**/*.jpg'), recursive=True)
        all_images.extend(images[:10])  # Get up to 10 images from each directory
    
    # Convert to relative paths
    dir_info['sample_image_paths'] = [os.path.relpath(img, data_path) for img in all_images[:10]]
    
    return dir_info


def check_image_paths(json_data, data_path, structure_info, check_first=-1):
    """
    Check if image paths in the JSON exist on disk.
    
    Args:
        json_data: Loaded JSON data
        data_path: Path to the dataset root directory
        structure_info: Dictionary with structure information
        check_first: Number of images to check first (-1 for all)
        
    Returns:
        path_info: Dictionary with path information
    """
    path_info = {
        'total_images': 0,
        'existing_images': 0,
        'missing_images': 0,
        'missing_paths': [],
        'path_patterns': defaultdict(int),
        'suggested_prefix': None,
        'suggested_fixes': {}
    }
    
    # Get image paths from JSON
    image_paths = []
    if structure_info['is_coco']:
        image_paths = [img.get('file_name', '') for img in json_data.get('images', [])]
    else:
        image_paths = list(json_data.keys())
    
    # Limit the number of images to check if needed
    if check_first > 0:
        image_paths = image_paths[:check_first]
    
    path_info['total_images'] = len(image_paths)
    
    # Check if images exist
    for img_path in tqdm(image_paths, desc="Checking image paths"):
        # Try different path combinations
        found = False
        
        # Direct path
        if os.path.exists(os.path.join(data_path, img_path)):
            found = True
            path_info['path_patterns']['direct'] += 1
        
        # With WIDER_train prefix
        elif os.path.exists(os.path.join(data_path, 'WIDER_train', img_path)):
            found = True
            path_info['path_patterns']['WIDER_train/'] += 1
            path_info['suggested_fixes'][img_path] = os.path.join('WIDER_train', img_path)
        
        # With images prefix
        elif os.path.exists(os.path.join(data_path, 'images', img_path)):
            found = True
            path_info['path_patterns']['images/'] += 1
            path_info['suggested_fixes'][img_path] = os.path.join('images', img_path)
        
        # With train prefix
        elif os.path.exists(os.path.join(data_path, 'train', img_path)):
            found = True
            path_info['path_patterns']['train/'] += 1
            path_info['suggested_fixes'][img_path] = os.path.join('train', img_path)
        
        # Try to find by filename only
        else:
            filename = os.path.basename(img_path)
            matches = glob.glob(os.path.join(data_path, '**', filename), recursive=True)
            if matches:
                found = True
                rel_path = os.path.relpath(matches[0], data_path)
                prefix = os.path.dirname(rel_path)
                if prefix:
                    path_info['path_patterns'][f'{prefix}/'] += 1
                    path_info['suggested_fixes'][img_path] = rel_path
                else:
                    path_info['path_patterns']['filename_only'] += 1
                    path_info['suggested_fixes'][img_path] = filename
        
        if found:
            path_info['existing_images'] += 1
        else:
            path_info['missing_images'] += 1
            path_info['missing_paths'].append(img_path)
    
    # Determine the most common path pattern
    if path_info['path_patterns']:
        most_common = max(path_info['path_patterns'].items(), key=lambda x: x[1])
        path_info['suggested_prefix'] = most_common[0]
    
    return path_info


def analyze_bbox_and_landmarks(json_data, structure_info):
    """
    Analyze bounding boxes and landmarks in the JSON data.
    
    Args:
        json_data: Loaded JSON data
        structure_info: Dictionary with structure information
        
    Returns:
        annotation_info: Dictionary with annotation information
    """
    annotation_info = {
        'bbox_format': 'unknown',
        'bbox_count': 0,
        'landmark_format': 'unknown',
        'landmark_count': 0,
        'has_visibility': False,
        'bbox_stats': {
            'min_width': float('inf'),
            'max_width': 0,
            'min_height': float('inf'),
            'max_height': 0,
            'avg_width': 0,
            'avg_height': 0
        },
        'landmark_stats': {
            'visible_count': 0,
            'invisible_count': 0,
            'visibility_by_point': [0, 0, 0, 0, 0]
        }
    }
    
    # Process annotations
    if structure_info['is_coco']:
        # COCO format
        annotations = json_data.get('annotations', [])
        annotation_info['bbox_count'] = len(annotations)
        
        # Check bbox format
        if annotations and 'bbox' in annotations[0]:
            bbox = annotations[0]['bbox']
            if len(bbox) == 4:
                # Check if it's [x, y, w, h] or [x1, y1, x2, y2]
                annotation_info['bbox_format'] = '[x, y, w, h]'  # COCO uses [x, y, w, h]
        
        # Check landmark format
        if annotations and 'keypoints' in annotations[0]:
            keypoints = annotations[0]['keypoints']
            annotation_info['landmark_count'] = len(annotations)
            
            if len(keypoints) % 3 == 0:
                annotation_info['landmark_format'] = '[x, y, v] * N'
                annotation_info['has_visibility'] = True
            elif len(keypoints) % 2 == 0:
                annotation_info['landmark_format'] = '[x, y] * N'
        
        # Calculate statistics
        total_width = 0
        total_height = 0
        
        for ann in annotations:
            if 'bbox' in ann:
                bbox = ann['bbox']
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    
                    annotation_info['bbox_stats']['min_width'] = min(annotation_info['bbox_stats']['min_width'], w)
                    annotation_info['bbox_stats']['max_width'] = max(annotation_info['bbox_stats']['max_width'], w)
                    annotation_info['bbox_stats']['min_height'] = min(annotation_info['bbox_stats']['min_height'], h)
                    annotation_info['bbox_stats']['max_height'] = max(annotation_info['bbox_stats']['max_height'], h)
                    
                    total_width += w
                    total_height += h
            
            if 'keypoints' in ann:
                keypoints = ann['keypoints']
                if len(keypoints) % 3 == 0:
                    for i in range(0, len(keypoints), 3):
                        x, y, v = keypoints[i:i+3]
                        if v > 0:
                            annotation_info['landmark_stats']['visible_count'] += 1
                            annotation_info['landmark_stats']['visibility_by_point'][i//3 % 5] += 1
                        else:
                            annotation_info['landmark_stats']['invisible_count'] += 1
        
        if annotation_info['bbox_count'] > 0:
            annotation_info['bbox_stats']['avg_width'] = total_width / annotation_info['bbox_count']
            annotation_info['bbox_stats']['avg_height'] = total_height / annotation_info['bbox_count']
    
    else:
        # Simple mapping format
        annotation_info['bbox_count'] = sum(len(faces) for faces in json_data.values())
        
        # Check bbox format
        for img_path, faces in json_data.items():
            if faces and 'bbox' in faces[0]:
                bbox = faces[0]['bbox']
                if len(bbox) == 4:
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        annotation_info['bbox_format'] = '[x1, y1, x2, y2]'
                    else:
                        annotation_info['bbox_format'] = '[x, y, w, h]'
                break
        
        # Check landmark format
        for img_path, faces in json_data.items():
            if faces and 'landmarks' in faces[0]:
                landmarks = faces[0]['landmarks']
                annotation_info['landmark_count'] = sum(1 for faces_list in json_data.values() 
                                                      for face in faces_list if 'landmarks' in face)
                
                if len(landmarks) % 3 == 0:
                    annotation_info['landmark_format'] = '[x, y, v] * N'
                    annotation_info['has_visibility'] = True
                elif len(landmarks) % 2 == 0:
                    annotation_info['landmark_format'] = '[x, y] * N'
                break
        
        # Calculate statistics
        total_width = 0
        total_height = 0
        
        for img_path, faces in json_data.items():
            for face in faces:
                if 'bbox' in face:
                    bbox = face['bbox']
                    if len(bbox) == 4:
                        if annotation_info['bbox_format'] == '[x, y, w, h]':
                            x, y, w, h = bbox
                        else:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        
                        annotation_info['bbox_stats']['min_width'] = min(annotation_info['bbox_stats']['min_width'], w)
                        annotation_info['bbox_stats']['max_width'] = max(annotation_info['bbox_stats']['max_width'], w)
                        annotation_info['bbox_stats']['min_height'] = min(annotation_info['bbox_stats']['min_height'], h)
                        annotation_info['bbox_stats']['max_height'] = max(annotation_info['bbox_stats']['max_height'], h)
                        
                        total_width += w
                        total_height += h
                
                if 'landmarks' in face:
                    landmarks = face['landmarks']
                    if annotation_info['landmark_format'] == '[x, y, v] * N':
                        for i in range(0, len(landmarks), 3):
                            x, y, v = landmarks[i:i+3]
                            if v > 0:
                                annotation_info['landmark_stats']['visible_count'] += 1
                                annotation_info['landmark_stats']['visibility_by_point'][i//3 % 5] += 1
                            else:
                                annotation_info['landmark_stats']['invisible_count'] += 1
                    else:
                        for i in range(0, len(landmarks), 2):
                            x, y = landmarks[i:i+2]
                            if x > 0 or y > 0:
                                annotation_info['landmark_stats']['visible_count'] += 1
                                annotation_info['landmark_stats']['visibility_by_point'][i//2 % 5] += 1
                            else:
                                annotation_info['landmark_stats']['invisible_count'] += 1
        
        if annotation_info['bbox_count'] > 0:
            annotation_info['bbox_stats']['avg_width'] = total_width / annotation_info['bbox_count']
            annotation_info['bbox_stats']['avg_height'] = total_height / annotation_info['bbox_count']
    
    return annotation_info


def fix_json_paths(json_data, structure_info, path_info):
    """
    Fix image paths in the JSON data.
    
    Args:
        json_data: Loaded JSON data
        structure_info: Dictionary with structure information
        path_info: Dictionary with path information
        
    Returns:
        fixed_json: JSON data with fixed paths
    """
    fixed_json = json_data.copy() if isinstance(json_data, dict) else json_data
    
    if not path_info['suggested_fixes']:
        logger.warning("No path fixes suggested. Returning original JSON.")
        return fixed_json
    
    if structure_info['is_coco']:
        # COCO format
        for img in fixed_json.get('images', []):
            file_name = img.get('file_name', '')
            if file_name in path_info['suggested_fixes']:
                img['file_name'] = path_info['suggested_fixes'][file_name]
    else:
        # Simple mapping format
        new_json = {}
        for img_path, faces in fixed_json.items():
            if img_path in path_info['suggested_fixes']:
                new_json[path_info['suggested_fixes'][img_path]] = faces
            else:
                new_json[img_path] = faces
        fixed_json = new_json
    
    return fixed_json


def print_summary(structure_info, dir_info, path_info, annotation_info, args):
    """Print a summary of the dataset analysis."""
    print("\n" + "="*80)
    print(" "*30 + "DATASET DEBUG SUMMARY")
    print("="*80)
    
    print("\n--- JSON STRUCTURE ---")
    print(f"Format: {structure_info['format']}")
    print(f"Is COCO format: {structure_info['is_coco']}")
    print(f"Image count: {structure_info['image_count']}")
    print(f"Annotation count: {structure_info['annotation_count']}")
    if structure_info['is_coco']:
        print(f"Category count: {structure_info['category_count']}")
    
    print("\nSample image paths from JSON:")
    for path in structure_info['sample_image_paths']:
        print(f"  - {path}")
    
    print(f"\nSample bbox format: {structure_info['sample_bbox_format']}")
    print(f"Sample keypoint format: {structure_info['sample_keypoint_format']}")
    
    print("\n--- DIRECTORY STRUCTURE ---")
    print(f"Data path exists: {dir_info['exists']}")
    print(f"Is directory: {dir_info['is_dir']}")
    print(f"Subdirectories: {', '.join(dir_info['subdirs'])}")
    
    print("\nImage directories:")
    for img_dir in dir_info['image_dirs']:
        print(f"  - {img_dir['path']}: {img_dir['image_count']} images, has subdirs: {img_dir['has_subdirs']}")
    
    print("\nSample image paths from disk:")
    for path in dir_info['sample_image_paths']:
        print(f"  - {path}")
    
    print("\n--- PATH ANALYSIS ---")
    print(f"Total images in JSON: {path_info['total_images']}")
    print(f"Existing images: {path_info['existing_images']} ({path_info['existing_images']/max(1, path_info['total_images'])*100:.2f}%)")
    print(f"Missing images: {path_info['missing_images']} ({path_info['missing_images']/max(1, path_info['total_images'])*100:.2f}%)")
    
    print("\nPath patterns found:")
    for pattern, count in path_info['path_patterns'].items():
        print(f"  - {pattern}: {count} images ({count/max(1, path_info['total_images'])*100:.2f}%)")
    
    print(f"\nSuggested path prefix: {path_info['suggested_prefix']}")
    
    if args.verbose and path_info['missing_paths']:
        print("\nSample missing paths:")
        for path in path_info['missing_paths'][:10]:
            print(f"  - {path}")
            if path in path_info['suggested_fixes']:
                print(f"    Suggested fix: {path_info['suggested_fixes'][path]}")
    
    print("\n--- ANNOTATION ANALYSIS ---")
    print(f"Bounding box format: {annotation_info['bbox_format']}")
    print(f"Bounding box count: {annotation_info['bbox_count']}")
    print(f"Landmark format: {annotation_info['landmark_format']}")
    print(f"Landmark count: {annotation_info['landmark_count']}")
    print(f"Has visibility flags: {annotation_info['has_visibility']}")
    
    print("\nBounding box statistics:")
    print(f"  - Min width: {annotation_info['bbox_stats']['min_width']:.2f} pixels")
    print(f"  - Max width: {annotation_info['bbox_stats']['max_width']:.2f} pixels")
    print(f"  - Min height: {annotation_info['bbox_stats']['min_height']:.2f} pixels")
    print(f"  - Max height: {annotation_info['bbox_stats']['max_height']:.2f} pixels")
    print(f"  - Average width: {annotation_info['bbox_stats']['avg_width']:.2f} pixels")
    print(f"  - Average height: {annotation_info['bbox_stats']['avg_height']:.2f} pixels")
    
    print("\nLandmark statistics:")
    total_landmarks = annotation_info['landmark_stats']['visible_count'] + annotation_info['landmark_stats']['invisible_count']
    print(f"  - Visible landmarks: {annotation_info['landmark_stats']['visible_count']} ({annotation_info['landmark_stats']['visible_count']/max(1, total_landmarks)*100:.2f}%)")
    print(f"  - Invisible landmarks: {annotation_info['landmark_stats']['invisible_count']} ({annotation_info['landmark_stats']['invisible_count']/max(1, total_landmarks)*100:.2f}%)")
    
    landmark_names = ['Right Eye', 'Left Eye', 'Nose', 'Right Mouth', 'Left Mouth']
    print("\nVisibility by landmark:")
    for i, name in enumerate(landmark_names):
        visibility = annotation_info['landmark_stats']['visibility_by_point'][i]
        print(f"  - {name}: {visibility} ({visibility/max(1, annotation_info['landmark_count'])*100:.2f}%)")
    
    print("\n--- RECOMMENDATIONS ---")
    if path_info['missing_images'] > 0:
        print("1. Path issues detected. Consider the following fixes:")
        if path_info['suggested_prefix'] != 'direct':
            print(f"   - Add '{path_info['suggested_prefix']}' prefix to image paths in JSON")
            print(f"   - Use the --fix-paths option to generate a fixed JSON file")
        else:
            print("   - The directory structure appears correct, but some images are missing")
            print("   - Check if the images are actually in the dataset")
    else:
        print("1. No path issues detected. The JSON paths match the directory structure.")
    
    if annotation_info['bbox_count'] != structure_info['annotation_count']:
        print(f"2. Bounding box count ({annotation_info['bbox_count']}) doesn't match annotation count ({structure_info['annotation_count']})")
        print("   - Check if all annotations have bounding boxes")
    
    if annotation_info['landmark_count'] != structure_info['annotation_count']:
        print(f"3. Landmark count ({annotation_info['landmark_count']}) doesn't match annotation count ({structure_info['annotation_count']})")
        print("   - Some annotations might be missing landmarks")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    args = parse_args()
    
    # Load JSON file
    ann_file_path = os.path.join(args.data_path, args.ann_file)
    logger.info(f"Loading JSON file: {ann_file_path}")
    json_data = load_json(ann_file_path)
    
    # Analyze JSON structure
    logger.info("Analyzing JSON structure")
    structure_info = analyze_json_structure(json_data)
    
    # Analyze directory structure
    logger.info("Analyzing directory structure")
    dir_info = analyze_directory_structure(args.data_path)
    
    # Check image paths
    logger.info("Checking image paths")
    path_info = check_image_paths(json_data, args.data_path, structure_info, args.check_first)
    
    # Analyze bounding boxes and landmarks
    logger.info("Analyzing annotations")
    annotation_info = analyze_bbox_and_landmarks(json_data, structure_info)
    
    # Print summary
    print_summary(structure_info, dir_info, path_info, annotation_info, args)
    
    # Fix JSON paths if requested
    if args.fix_paths:
        logger.info("Fixing JSON paths")
        fixed_json = fix_json_paths(json_data, structure_info, path_info)
        
        # Save fixed JSON
        output_path = os.path.join(args.data_path, args.output_file)
        with open(output_path, 'w') as f:
            json.dump(fixed_json, f, indent=2)
        
        logger.info(f"Saved fixed JSON to {output_path}")
        print(f"\nFixed JSON saved to: {output_path}")
    
    logger.info("Dataset debug completed")


if __name__ == '__main__':
    main()
