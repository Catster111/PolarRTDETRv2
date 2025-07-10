"""
WiderFace dataset with 5 landmarks support for Polar-RTDETRv2.

This module provides dataset implementation for the WiderFace dataset
with additional support for 5 facial landmarks. It includes:
- Custom dataset class for WiderFace with landmarks
- Polar coordinate transformation
- Data augmentation with landmark preservation
- Utilities for parsing annotations
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import json
import logging
from pathlib import Path
import random
import math
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from .transforms import Compose


logger = logging.getLogger(__name__)


def cart2polar(x, y, center_x, center_y):
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        x: x-coordinate
        y: y-coordinate
        center_x: x-coordinate of the center point
        center_y: y-coordinate of the center point
        
    Returns:
        r: radius (distance from center)
        theta: angle in radians
    """
    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return r, theta


def polar2cart(r, theta, center_x, center_y):
    """
    Convert polar coordinates to Cartesian coordinates.
    
    Args:
        r: radius (distance from center)
        theta: angle in radians
        center_x: x-coordinate of the center point
        center_y: y-coordinate of the center point
        
    Returns:
        x: x-coordinate
        y: y-coordinate
    """
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    return x, y


def bbox2polar(bbox, img_width, img_height):
    """
    Convert bounding box to polar coordinates.
    
    Args:
        bbox: List of [x1, y1, x2, y2] in absolute coordinates
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        polar_bbox: List of [r1, theta1, r2, theta2] in polar coordinates
    """
    center_x = img_width / 2
    center_y = img_height / 2
    
    x1, y1, x2, y2 = bbox
    
    # Convert corners to polar coordinates
    r1, theta1 = cart2polar(x1, y1, center_x, center_y)
    r2, theta2 = cart2polar(x2, y2, center_x, center_y)
    
    return [r1, theta1, r2, theta2]


def landmarks2polar(landmarks, img_width, img_height):
    """
    Convert facial landmarks to polar coordinates.
    
    Args:
        landmarks: List of [x1, y1, x2, y2, ..., x5, y5] in absolute coordinates
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        polar_landmarks: List of [r1, theta1, r2, theta2, ..., r5, theta5] in polar coordinates
    """
    center_x = img_width / 2
    center_y = img_height / 2
    
    polar_landmarks = []
    for i in range(0, len(landmarks), 2):
        x, y = landmarks[i], landmarks[i+1]
        r, theta = cart2polar(x, y, center_x, center_y)
        polar_landmarks.extend([r, theta])
    
    return polar_landmarks


def parse_widerface_annotation(annotation_file: str) -> Dict[str, List]:
    """
    Parse WiderFace annotation file.
    
    Args:
        annotation_file: Path to the annotation file
        
    Returns:
        annotations: Dictionary mapping image paths to lists of face annotations
    """
    # ------------------------------------------------------------------
    # New: JSON support. If the file is *.json we expect a mapping:
    # {
    #   "relative/path/to/img_1.jpg": [
    #        {"bbox": [x1,y1,x2,y2] OR [x,y,w,h],
    #         "landmarks": [x1,y1, ..., x5,y5]}
    #   ],
    #   ...
    # }
    # ------------------------------------------------------------------
    if annotation_file.lower().endswith(".json"):
        with open(annotation_file, "r") as f:
            raw_ann = json.load(f)

        # ------------------------------------------------------------------
        # Case A: Simple mapping  {image_path: [faces,...]}
        # ------------------------------------------------------------------
        if all(isinstance(v, list) for v in raw_ann.values()):
            annotations: Dict[str, List] = {}
            for img_path, faces in raw_ann.items():
                annotations[img_path] = []
                for face in faces:
                    bbox = face.get("bbox", [])
                    # Convert bbox to [x1,y1,x2,y2] if provided as [x,y,w,h]
                    if len(bbox) == 4:
                        x1, y1, w, h = bbox
                        bbox = [x1, y1, x1 + w, y1 + h]
                    ann_dict = {
                        "bbox": bbox,
                        "blur": face.get("blur", 0),
                        "expression": face.get("expression", 0),
                        "illumination": face.get("illumination", 0),
                        "invalid": face.get("invalid", 0),
                        "occlusion": face.get("occlusion", 0),
                        "pose": face.get("pose", 0),
                        "landmarks": face.get("landmarks", []),
                    }
                    annotations[img_path].append(ann_dict)
            return annotations

        # ------------------------------------------------------------------
        # Case B: COCO-style dict with "images" and "annotations" arrays
        # ------------------------------------------------------------------
        if "images" in raw_ann and "annotations" in raw_ann:
            id_to_file: Dict[int, str] = {
                img["id"]: img["file_name"] for img in raw_ann["images"]
            }

            annotations: Dict[str, List] = {fn: [] for fn in id_to_file.values()}

            for ann in raw_ann["annotations"]:
                img_id = ann["image_id"]
                file_name = id_to_file.get(img_id)
                if file_name is None:
                    continue

                # COCO bbox is [x, y, w, h] -> convert to [x1,y1,x2,y2]
                x, y, w, h = ann["bbox"]
                bbox = [x, y, x + w, y + h]

                # COCO keypoints: 15 values [x,y,v]*5
                kp = ann.get("keypoints", [])
                landmarks: List[float] = []
                if len(kp) == 15:
                    for j in range(0, 15, 3):
                        lx, ly, vis = kp[j], kp[j + 1], kp[j + 2]
                        # if not visible set 0
                        if vis == 0:
                            landmarks.extend([0.0, 0.0])
                        else:
                            landmarks.extend([lx, ly])
                # Ensure length 10
                if len(landmarks) != 10:
                    landmarks = [0.0] * 10

                annotations[file_name].append(
                    {
                        "bbox": bbox,
                        "blur": 0,
                        "expression": 0,
                        "illumination": 0,
                        "invalid": ann.get("iscrowd", 0),
                        "occlusion": 0,
                        "pose": 0,
                        "landmarks": landmarks,
                    }
                )

            # Remove images with no faces
            annotations = {k: v for k, v in annotations.items() if v}
            return annotations

        # Fallback: unknown JSON structure
        logger.error(
            "Unsupported JSON annotation structure. Expected mapping or COCO style."
        )
        return {}

    # ------------------------------------------------------------------
    # Legacy WIDER txt format parsing (unchanged)
    # ------------------------------------------------------------------
    annotations = {}
    current_image = None
    face_count = 0

    with open(annotation_file, "r") as f:
        lines = f.readlines()

    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()

        if line.endswith(".jpg") or line.endswith(".png"):
            # This is an image file path
            current_image = line
            annotations[current_image] = []
            line_idx += 1

            # Next line contains the number of faces
            face_count = int(lines[line_idx].strip())
            line_idx += 1

            # Parse each face annotation
            for _ in range(face_count):
                if line_idx < len(lines):
                    face_anno = lines[line_idx].strip().split()

                    # Basic bbox: [x, y, width, height]
                    x = float(face_anno[0])
                    y = float(face_anno[1])
                    w = float(face_anno[2])
                    h = float(face_anno[3])

                    # Convert to [x1, y1, x2, y2] format
                    bbox = [x, y, x + w, y + h]

                    # Create annotation with bbox
                    annotation = {
                        "bbox": bbox,
                        "blur": int(face_anno[4]),
                        "expression": int(face_anno[5]),
                        "illumination": int(face_anno[6]),
                        "invalid": int(face_anno[7]),
                        "occlusion": int(face_anno[8]),
                        "pose": int(face_anno[9]),
                    }

                    annotations[current_image].append(annotation)
                    line_idx += 1
        else:
            line_idx += 1

    return annotations


def parse_landmarks_annotation(landmarks_file: str) -> Dict[str, List[List[float]]]:
    """
    Parse landmarks annotation file.
    
    Args:
        landmarks_file: Path to the landmarks annotation file
        
    Returns:
        landmarks: Dictionary mapping image paths to lists of landmark annotations
    """
    landmarks = {}
    
    # Check if the landmarks file exists
    if not os.path.exists(landmarks_file):
        logger.warning(f"Landmarks file {landmarks_file} not found.")
        return landmarks
    
    # Try to parse as JSON first (preferred format)
    try:
        with open(landmarks_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        pass
    
    # Fall back to custom format parsing
    with open(landmarks_file, 'r') as f:
        lines = f.readlines()
    
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        
        if line.endswith('.jpg') or line.endswith('.png'):
            # This is an image file path
            current_image = line
            landmarks[current_image] = []
            line_idx += 1
            
            # Next line contains the number of faces
            face_count = int(lines[line_idx].strip())
            line_idx += 1
            
            # Parse each face's landmarks
            for _ in range(face_count):
                if line_idx < len(lines):
                    landmark_line = lines[line_idx].strip().split()
                    
                    # Each landmark has 10 values (5 landmarks, each with x and y)
                    face_landmarks = [float(val) for val in landmark_line[:10]]
                    
                    landmarks[current_image].append(face_landmarks)
                    line_idx += 1
        else:
            line_idx += 1
    
    return landmarks


def merge_annotations_with_landmarks(annotations: Dict, landmarks: Dict) -> Dict:
    """
    Merge face annotations with landmark annotations.
    
    Args:
        annotations: Dictionary of face annotations
        landmarks: Dictionary of landmark annotations
        
    Returns:
        merged: Dictionary with merged annotations
    """
    merged = {}
    
    for image_path, faces in annotations.items():
        merged[image_path] = []
        
        # Get landmarks for this image if available
        image_landmarks = landmarks.get(image_path, [])
        
        # Ensure we have the same number of faces and landmarks
        if len(faces) != len(image_landmarks) and len(image_landmarks) > 0:
            logger.warning(
                f"Mismatch between number of faces ({len(faces)}) and "
                f"landmarks ({len(image_landmarks)}) for {image_path}"
            )
        
        # Merge face annotations with landmarks
        for i, face in enumerate(faces):
            face_with_landmarks = face.copy()
            
            # Add landmarks if available
            if i < len(image_landmarks):
                face_with_landmarks['landmarks'] = image_landmarks[i]
            else:
                # If no landmarks, add dummy values
                face_with_landmarks['landmarks'] = [0.0] * 10
            
            merged[image_path].append(face_with_landmarks)
    
    return merged


class WiderFaceLandmarksDataset(Dataset):
    """
    WiderFace dataset with 5 landmarks support.
    
    This dataset class loads WiderFace images and annotations,
    including 5 facial landmarks per face. It supports:
    - Polar coordinate transformation
    - Data augmentation
    - Training and evaluation modes
    """
    
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        landmarks_file: Optional[str] = None,
        transforms: Optional[Callable] = None,
        is_train: bool = True,
        use_polar: bool = True,
        filter_invalid: bool = True,
        min_face_size: int = 8,
        cache_mode: bool = False,
        exclude_images: List[str] = None,
        image_prefix: str = ""
    ):
        """
        Initialize the WiderFace dataset with landmarks.
        
        Args:
            img_folder: Path to the image folder
            ann_file: Path to the annotation file
            landmarks_file: Path to the landmarks file (optional)
            transforms: Transformations to apply to images and annotations
            is_train: Whether in training mode
            use_polar: Whether to use polar coordinates
            filter_invalid: Whether to filter out invalid faces
            min_face_size: Minimum face size to include
            cache_mode: Whether to cache images in memory
            exclude_images: List of image paths to exclude
            image_prefix: Prefix to add to image paths from JSON (e.g., 'WIDER_train/images/')
        """
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.landmarks_file = landmarks_file
        self.transforms = transforms
        self.is_train = is_train
        self.use_polar = use_polar
        self.filter_invalid = filter_invalid
        self.min_face_size = min_face_size
        self.cache_mode = cache_mode
        self.exclude_images = exclude_images if exclude_images else []
        self.image_prefix = image_prefix
        
        # Parse annotations and landmarks
        self.annotations = parse_widerface_annotation(ann_file)
        
        # Parse landmarks if provided
        if landmarks_file and os.path.exists(landmarks_file):
            self.landmarks = parse_landmarks_annotation(landmarks_file)
            # Merge annotations with landmarks
            self.annotations = merge_annotations_with_landmarks(
                self.annotations, self.landmarks
            )
        
        # Create list of valid image paths and annotations
        self.ids = []
        for img_path, faces in self.annotations.items():
            # Skip excluded images
            if img_path in self.exclude_images:
                continue
                
            # Filter faces if needed
            valid_faces = []
            for face in faces:
                bbox = face['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Skip invalid or small faces during training
                if self.is_train and self.filter_invalid:
                    if (face.get('invalid', 0) == 1 or 
                        width < self.min_face_size or 
                        height < self.min_face_size):
                        continue
                
                valid_faces.append(face)
            
            # Skip images with no valid faces during training
            if self.is_train and len(valid_faces) == 0:
                continue
            
            # Add image to dataset
            self.ids.append((img_path, valid_faces))
        
        # Cache for images
        self.cache = {} if cache_mode else None
        
        logger.info(
            f"Loaded {len(self.ids)} images with valid faces "
            f"from WiderFace dataset"
        )
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            sample: Dictionary containing image, targets, and metadata
        """
        img_path, faces = self.ids[idx]
        
        # Load image
        img = self.get_image(img_path)
        img_width, img_height = img.size
        
        # Prepare annotations
        boxes = []
        labels = []
        landmarks = []
        
        for face in faces:
            # Get bbox
            bbox = face['bbox']
            boxes.append(bbox)
            
            # Face class is always 1
            labels.append(1)
            
            # Get landmarks
            face_landmarks = face.get('landmarks', [0.0] * 10)
            landmarks.append(face_landmarks)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        landmarks = torch.as_tensor(landmarks, dtype=torch.float32)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'landmarks': landmarks,
            'image_id': idx,
            'orig_size': torch.as_tensor([img_height, img_width]),
            'size': torch.as_tensor([img_height, img_width]),
            'file_name': img_path,
        }
        
        # Apply transformations
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        # Convert to polar coordinates if needed
        if self.use_polar and len(boxes) > 0:
            polar_boxes = []
            polar_landmarks = []
            
            # Get current image size (may have changed after transforms)
            curr_height, curr_width = target['size'].tolist()
            
            # Convert boxes to polar
            for box in target['boxes'].tolist():
                polar_box = bbox2polar(box, curr_width, curr_height)
                polar_boxes.append(polar_box)
            
            # Convert landmarks to polar
            for lm in target['landmarks'].tolist():
                polar_lm = landmarks2polar(lm, curr_width, curr_height)
                polar_landmarks.append(polar_lm)
            
            # Update target with polar coordinates
            target['polar_boxes'] = torch.as_tensor(polar_boxes, dtype=torch.float32)
            target['polar_landmarks'] = torch.as_tensor(polar_landmarks, dtype=torch.float32)
        
        return img, target
    
    def get_image(self, img_path):
        """
        Load an image from disk or cache.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            img: PIL Image
        """
        # Try to get from cache first
        if self.cache_mode and img_path in self.cache:
            return self.cache[img_path]
        
        # Apply image prefix if provided (e.g., 'WIDER_train/images/')
        prefixed_path = os.path.join(self.image_prefix, img_path) if self.image_prefix else img_path
        
        # Load from disk
        full_path = os.path.join(self.img_folder, prefixed_path)
        img = Image.open(full_path).convert('RGB')
        
        # Cache if needed
        if self.cache_mode:
            self.cache[img_path] = img
        
        return img


def collate_fn(batch):
    """
    Collate function for batching samples.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        images: Tensor of batched images
        targets: List of target dictionaries
    """
    images, targets = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    return images, targets


def build_widerface(
    image_set: str,
    args: Any
) -> Tuple[Dataset, int]:
    """
    Build WiderFace dataset.
    
    Args:
        image_set: 'train' or 'val'
        args: Configuration arguments
        
    Returns:
        dataset: WiderFace dataset
        num_classes: Number of classes (always 2: background and face)
    """
    root = Path(args.widerface_path)
    assert root.exists(), f"WiderFace path {root} does not exist"
    
    # ---------------------------
    #   Paths & Prefix handling
    # ---------------------------
    # Images live under root / <image_prefix> / <json-path>
    # Example: json has '0--Parade/xxx.jpg'
    #   widerface_path = '/data/widerface'
    #   image_prefix   = 'WIDER_train/images'
    #   final path     = /data/widerface/WIDER_train/images/0--Parade/xxx.jpg
    image_prefix = getattr(args, "image_prefix", "")

    # Set annotation file paths based on split
    if image_set == 'train':
        img_folder = root  # base path; prefix will be applied later
        ann_file = root / 'annotations' / 'train_wider_face.json'
        landmarks_file = None
    elif image_set == 'val':
        img_folder = root
        ann_file = root / 'annotations' / 'val_wider_face.json'
        landmarks_file = None
    else:
        raise ValueError(f"Unknown image_set: {image_set}")
    
    # Check if files exist
    assert img_folder.exists(), f"Image folder {img_folder} does not exist"
    assert ann_file.exists(), f"Annotation file {ann_file} does not exist"
    
    # Create transforms
    if image_set == 'train':
        transforms = Compose([
            # Add your training transforms here
        ])
    else:
        transforms = Compose([
            # Add your validation transforms here
        ])
    
    # Create dataset
    dataset = WiderFaceLandmarksDataset(
        img_folder=str(img_folder),
        ann_file=str(ann_file),
        landmarks_file=str(landmarks_file) if (landmarks_file and landmarks_file.exists()) else None,
        transforms=transforms,
        is_train=(image_set == 'train'),
        use_polar=args.use_polar,
        filter_invalid=args.filter_invalid if hasattr(args, 'filter_invalid') else True,
        min_face_size=args.min_face_size if hasattr(args, 'min_face_size') else 8,
        cache_mode=args.cache_mode if hasattr(args, 'cache_mode') else False,
        image_prefix=image_prefix,
    )
    
    # Face detection has 2 classes: background (0) and face (1)
    return dataset, 2
