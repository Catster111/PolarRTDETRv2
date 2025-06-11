"""
WIDER FACE Dataset Utilities
Utilities for downloading, processing and preparing WIDER FACE dataset
"""

import os
import json
import requests
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

def download_wider_face(download_dir: str = "./dataset/wider_face") -> None:
    """Download WIDER FACE dataset
    
    Args:
        download_dir: Directory to download and extract dataset
    """
    
    # Create download directory
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    # WIDER FACE URLs
    urls = {
        "train_images": "https://drive.google.com/uc?id=0B6eKvaijfFUDQUUwd21EckhUbWs",
        "val_images": "https://drive.google.com/uc?id=0B6eKvaijfFUDd3dIRmpvSk8tLUE", 
        "test_images": "https://drive.google.com/uc?id=0B6eKvaijfFUDbW4tdGpaYjgzZkU",
        "annotations": "https://drive.google.com/uc?id=0B6eKvaijfFUDOUNFcHRqcHZYHWs"
    }
    
    print("Note: WIDER FACE dataset requires manual download from:")
    print("http://shuoyang1213.me/WIDERFACE/")
    print("\nPlease download the following files to", download_dir)
    for name, url in urls.items():
        print(f"- {name}: {url}")
    
    print(f"\nAfter downloading, extract all files to {download_dir}")
    print("Expected structure:")
    print(f"{download_dir}/")
    print("├── WIDER_train/")
    print("├── WIDER_val/") 
    print("├── WIDER_test/")
    print("└── wider_face_split/")


def parse_wider_face_annotations(
    annotation_file: str,
    image_dir: str,
    output_file: str,
    min_face_size: int = 10
) -> Dict:
    """Parse WIDER FACE annotation file to standard format
    
    Args:
        annotation_file: Path to WIDER FACE annotation file
        image_dir: Directory containing images
        output_file: Output JSON file path
        min_face_size: Minimum face size to include
        
    Returns:
        Dictionary with parsed annotations
    """
    
    annotations = {}
    valid_images = 0
    total_faces = 0
    
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    pbar = tqdm(total=len(lines), desc="Parsing annotations")
    
    while i < len(lines):
        # Image path
        image_path = lines[i].strip()
        i += 1
        pbar.update(1)
        
        # Check if image exists
        full_image_path = os.path.join(image_dir, image_path)
        if not os.path.exists(full_image_path):
            # Skip number of faces and face data
            if i < len(lines):
                num_faces = int(lines[i].strip())
                i += 1 + max(num_faces, 1)  # Skip face annotations
                pbar.update(1 + max(num_faces, 1))
            continue
        
        # Number of faces
        num_faces = int(lines[i].strip())
        i += 1
        pbar.update(1)
        
        faces = []
        valid_faces = 0
        
        # Process each face
        for j in range(max(num_faces, 1)):  # At least read one line even if num_faces=0
            if i >= len(lines):
                break
                
            line = lines[i].strip()
            i += 1
            pbar.update(1)
            
            if num_faces == 0:
                # No faces in image, but still has one line to skip
                break
            
            # Parse face annotation
            parts = line.split()
            if len(parts) >= 4:
                x, y, w, h = map(int, parts[:4])
                
                # Additional attributes if available
                blur = int(parts[4]) if len(parts) > 4 else 0
                expression = int(parts[5]) if len(parts) > 5 else 0
                illumination = int(parts[6]) if len(parts) > 6 else 0
                invalid = int(parts[7]) if len(parts) > 7 else 0
                occlusion = int(parts[8]) if len(parts) > 8 else 0
                pose = int(parts[9]) if len(parts) > 9 else 0
                
                # Filter out invalid or too small faces
                if invalid == 0 and w >= min_face_size and h >= min_face_size:
                    face_data = {
                        'bbox': [x, y, w, h],
                        'blur': blur,
                        'expression': expression,
                        'illumination': illumination,
                        'occlusion': occlusion,
                        'pose': pose,
                        'landmarks': None  # Will be filled later if available
                    }
                    faces.append(face_data)
                    valid_faces += 1
        
        # Only include images with valid faces
        if valid_faces > 0:
            # Get image dimensions
            try:
                with Image.open(full_image_path) as img:
                    img_width, img_height = img.size
                
                annotations[image_path] = {
                    'image_width': img_width,
                    'image_height': img_height,
                    'faces': faces
                }
                valid_images += 1
                total_faces += valid_faces
                
            except Exception as e:
                print(f"Error loading image {full_image_path}: {e}")
                continue
    
    pbar.close()
    
    # Save annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Processed {valid_images} images with {total_faces} faces")
    print(f"Saved annotations to {output_file}")
    
    return annotations


def add_synthetic_landmarks(
    annotations: Dict,
    output_file: str,
    landmark_noise: float = 0.05
) -> Dict:
    """Add synthetic 5-point landmarks to face annotations
    
    This is a placeholder for when real landmark annotations are not available.
    In practice, you should use real landmark annotations or a landmark detector.
    
    Args:
        annotations: Face annotations dictionary
        output_file: Output file path
        landmark_noise: Noise level for synthetic landmarks
        
    Returns:
        Updated annotations with landmarks
    """
    
    print("Adding synthetic landmarks (for testing purposes only)")
    print("In production, use real landmark annotations!")
    
    for image_path, image_data in tqdm(annotations.items(), desc="Adding landmarks"):
        for face_data in image_data['faces']:
            x, y, w, h = face_data['bbox']
            
            # Generate synthetic 5-point landmarks within face bbox
            # Points: left_eye, right_eye, nose, left_mouth, right_mouth
            landmarks = []
            
            # Left eye (relative position in face)
            left_eye_x = x + w * (0.3 + np.random.normal(0, landmark_noise))
            left_eye_y = y + h * (0.35 + np.random.normal(0, landmark_noise))
            landmarks.extend([left_eye_x, left_eye_y])
            
            # Right eye
            right_eye_x = x + w * (0.7 + np.random.normal(0, landmark_noise))
            right_eye_y = y + h * (0.35 + np.random.normal(0, landmark_noise))
            landmarks.extend([right_eye_x, right_eye_y])
            
            # Nose
            nose_x = x + w * (0.5 + np.random.normal(0, landmark_noise))
            nose_y = y + h * (0.55 + np.random.normal(0, landmark_noise))
            landmarks.extend([nose_x, nose_y])
            
            # Left mouth corner
            left_mouth_x = x + w * (0.35 + np.random.normal(0, landmark_noise))
            left_mouth_y = y + h * (0.75 + np.random.normal(0, landmark_noise))
            landmarks.extend([left_mouth_x, left_mouth_y])
            
            # Right mouth corner
            right_mouth_x = x + w * (0.65 + np.random.normal(0, landmark_noise))
            right_mouth_y = y + h * (0.75 + np.random.normal(0, landmark_noise))
            landmarks.extend([right_mouth_x, right_mouth_y])
            
            face_data['landmarks'] = landmarks
    
    # Save updated annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Saved annotations with synthetic landmarks to {output_file}")
    return annotations


def create_coco_format_annotations(
    wider_annotations: Dict,
    output_file: str,
    category_name: str = "face"
) -> Dict:
    """Convert WIDER FACE annotations to COCO format
    
    Args:
        wider_annotations: WIDER FACE annotations
        output_file: Output COCO JSON file
        category_name: Category name for faces
        
    Returns:
        COCO format annotations
    """
    
    coco_data = {
        "info": {
            "description": "WIDER FACE Dataset in COCO format",
            "version": "1.0",
            "year": 2023
        },
        "categories": [
            {
                "id": 1,
                "name": category_name,
                "supercategory": "person"
            }
        ],
        "images": [],
        "annotations": []
    }
    
    image_id = 1
    annotation_id = 1
    
    for image_path, image_data in tqdm(wider_annotations.items(), desc="Converting to COCO format"):
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": image_path,
            "width": image_data['image_width'],
            "height": image_data['image_height']
        }
        coco_data["images"].append(image_info)
        
        # Add face annotations
        for face_data in image_data['faces']:
            x, y, w, h = face_data['bbox']
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            }
            
            # Add landmarks if available
            if face_data.get('landmarks'):
                annotation['keypoints'] = face_data['landmarks'] + [2] * 5  # Visibility = 2 (visible)
                annotation['num_keypoints'] = 5
            
            # Add WIDER FACE specific attributes
            annotation['attributes'] = {
                'blur': face_data.get('blur', 0),
                'expression': face_data.get('expression', 0),
                'illumination': face_data.get('illumination', 0),
                'occlusion': face_data.get('occlusion', 0),
                'pose': face_data.get('pose', 0)
            }
            
            coco_data["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    # Save COCO format annotations
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Saved COCO format annotations to {output_file}")
    return coco_data


def verify_dataset(dataset_root: str) -> bool:
    """Verify WIDER FACE dataset structure and files
    
    Args:
        dataset_root: Root directory of WIDER FACE dataset
        
    Returns:
        True if dataset is valid
    """
    
    required_dirs = [
        'WIDER_train/images',
        'WIDER_val/images', 
        'wider_face_split'
    ]
    
    required_files = [
        'wider_face_split/wider_face_train_bbx_gt.txt',
        'wider_face_split/wider_face_val_bbx_gt.txt'
    ]
    
    print("Verifying WIDER FACE dataset structure...")
    
    # Check directories
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_root, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing directory: {full_path}")
            return False
        print(f"✅ Found directory: {dir_path}")
    
    # Check files
    for file_path in required_files:
        full_path = os.path.join(dataset_root, file_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing file: {full_path}")
            return False
        print(f"✅ Found file: {file_path}")
    
    # Count images
    train_images = count_images(os.path.join(dataset_root, 'WIDER_train/images'))
    val_images = count_images(os.path.join(dataset_root, 'WIDER_val/images'))
    
    print(f"📊 Training images: {train_images}")
    print(f"📊 Validation images: {val_images}")
    
    if train_images == 0 or val_images == 0:
        print("❌ No images found in dataset")
        return False
    
    print("✅ Dataset structure verified successfully!")
    return True


def count_images(directory: str) -> int:
    """Count number of images in directory recursively"""
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                count += 1
    return count


def create_dataset_split(
    annotations: Dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[Dict, Dict]:
    """Split dataset into train and validation sets
    
    Args:
        annotations: Full annotations dictionary
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_annotations, val_annotations)
    """
    
    np.random.seed(random_seed)
    
    image_paths = list(annotations.keys())
    np.random.shuffle(image_paths)
    
    n_train = int(len(image_paths) * train_ratio)
    
    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:]
    
    train_annotations = {path: annotations[path] for path in train_paths}
    val_annotations = {path: annotations[path] for path in val_paths}
    
    print(f"Split dataset: {len(train_annotations)} train, {len(val_annotations)} val")
    
    return train_annotations, val_annotations


def prepare_wider_face_dataset(dataset_root: str) -> None:
    """Complete pipeline to prepare WIDER FACE dataset
    
    Args:
        dataset_root: Root directory containing WIDER FACE dataset
    """
    
    print("🚀 Starting WIDER FACE dataset preparation...")
    
    # Verify dataset structure
    if not verify_dataset(dataset_root):
        print("❌ Dataset verification failed!")
        return
    
    # Create output directory
    output_dir = os.path.join(dataset_root, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process training set
    print("\n📝 Processing training annotations...")
    train_annotations = parse_wider_face_annotations(
        annotation_file=os.path.join(dataset_root, 'wider_face_split/wider_face_train_bbx_gt.txt'),
        image_dir=os.path.join(dataset_root, 'WIDER_train/images'),
        output_file=os.path.join(output_dir, 'train_annotations.json')
    )
    
    # Process validation set
    print("\n📝 Processing validation annotations...")
    val_annotations = parse_wider_face_annotations(
        annotation_file=os.path.join(dataset_root, 'wider_face_split/wider_face_val_bbx_gt.txt'),
        image_dir=os.path.join(dataset_root, 'WIDER_val/images'),
        output_file=os.path.join(output_dir, 'val_annotations.json')
    )
    
    # Add synthetic landmarks (replace with real landmarks if available)
    print("\n🎯 Adding synthetic landmarks...")
    train_annotations = add_synthetic_landmarks(
        train_annotations,
        os.path.join(output_dir, 'train_annotations_with_landmarks.json')
    )
    
    val_annotations = add_synthetic_landmarks(
        val_annotations,
        os.path.join(output_dir, 'val_annotations_with_landmarks.json')
    )
    
    # Create COCO format annotations
    print("\n🔄 Converting to COCO format...")
    create_coco_format_annotations(
        train_annotations,
        os.path.join(output_dir, 'train_coco_format.json')
    )
    
    create_coco_format_annotations(
        val_annotations,
        os.path.join(output_dir, 'val_coco_format.json')
    )
    
    # Create symlinks for easy access
    print("\n🔗 Creating convenient symlinks...")
    
    # Symlink for landmarks annotations
    landmark_dir = os.path.join(dataset_root, 'wider_face_split')
    
    train_landmark_src = os.path.join(output_dir, 'train_annotations_with_landmarks.json')
    train_landmark_dst = os.path.join(landmark_dir, 'wider_face_train_landmarks.json')
    
    val_landmark_src = os.path.join(output_dir, 'val_annotations_with_landmarks.json') 
    val_landmark_dst = os.path.join(landmark_dir, 'wider_face_val_landmarks.json')
    
    # Remove existing symlinks
    for dst in [train_landmark_dst, val_landmark_dst]:
        if os.path.exists(dst):
            os.remove(dst)
    
    # Create new symlinks
    os.symlink(os.path.abspath(train_landmark_src), train_landmark_dst)
    os.symlink(os.path.abspath(val_landmark_src), val_landmark_dst)
    
    print("✅ Dataset preparation completed!")
    print(f"📁 Processed files saved to: {output_dir}")
    print(f"🔗 Landmark files linked in: {landmark_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WIDER FACE Dataset Utilities")
    parser.add_argument('--dataset_root', type=str, default='./dataset/wider_face',
                       help='Root directory of WIDER FACE dataset')
    parser.add_argument('--download', action='store_true',
                       help='Show download instructions')
    parser.add_argument('--prepare', action='store_true', 
                       help='Prepare dataset for training')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset structure')
    
    args = parser.parse_args()
    
    if args.download:
        download_wider_face(args.dataset_root)
    
    if args.verify:
        verify_dataset(args.dataset_root)
    
    if args.prepare:
        prepare_wider_face_dataset(args.dataset_root)
    
    if not any([args.download, args.prepare, args.verify]):
        print("Use --download, --verify, or --prepare to perform actions")
        print("Run with --help for more information")