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

from polar_datasets.transforms import Compose, ToTensor, Normalize


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


def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        normalized_angle: Normalized angle in [-pi, pi] range
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def bbox_to_polar(bbox, img_width, img_height, normalize_radius=True):
    """
    Convert bounding box from Cartesian to polar coordinates.
    
    Args:
        bbox: Bounding box in format [x1, y1, x2, y2]
        img_width: Image width
        img_height: Image height
        normalize_radius: Whether to normalize radius by image diagonal
        
    Returns:
        polar_bbox: Bounding box in polar coordinates [r1, theta1, r2, theta2]
    """
    x1, y1, x2, y2 = bbox
    
    # Image center as reference point
    center_x = img_width / 2
    center_y = img_height / 2
    
    # Convert corners to polar coordinates
    r1, theta1 = cart2polar(x1, y1, center_x, center_y)
    r2, theta2 = cart2polar(x2, y2, center_x, center_y)
    
    # Normalize radius by image diagonal if requested
    if normalize_radius:
        diagonal = np.sqrt(img_width**2 + img_height**2)
        r1 /= diagonal
        r2 /= diagonal
    
    return [r1, theta1, r2, theta2]


def polar_to_bbox(polar_bbox, img_width, img_height, normalize_radius=True):
    """
    Convert bounding box from polar to Cartesian coordinates.
    
    Args:
        polar_bbox: Bounding box in polar coordinates [r1, theta1, r2, theta2]
        img_width: Image width
        img_height: Image height
        normalize_radius: Whether radius was normalized by image diagonal
        
    Returns:
        bbox: Bounding box in format [x1, y1, x2, y2]
    """
    r1, theta1, r2, theta2 = polar_bbox
    
    # Image center as reference point
    center_x = img_width / 2
    center_y = img_height / 2
    
    # Denormalize radius if it was normalized
    if normalize_radius:
        diagonal = np.sqrt(img_width**2 + img_height**2)
        r1 *= diagonal
        r2 *= diagonal
    
    # Convert polar coordinates to Cartesian
    x1, y1 = polar2cart(r1, theta1, center_x, center_y)
    x2, y2 = polar2cart(r2, theta2, center_x, center_y)
    
    return [x1, y1, x2, y2]


def landmarks_to_polar(landmarks, img_width, img_height, normalize_radius=True):
    """
    Convert landmarks from Cartesian to polar coordinates.
    
    Args:
        landmarks: Landmarks in format [x1, y1, x2, y2, ..., xn, yn]
        img_width: Image width
        img_height: Image height
        normalize_radius: Whether to normalize radius by image diagonal
        
    Returns:
        polar_landmarks: Landmarks in polar coordinates [r1, theta1, r2, theta2, ..., rn, thetan]
    """
    # Image center as reference point
    center_x = img_width / 2
    center_y = img_height / 2
    
    # Diagonal for normalization
    diagonal = np.sqrt(img_width**2 + img_height**2) if normalize_radius else 1.0
    
    polar_landmarks = []
    for i in range(0, len(landmarks), 2):
        x, y = landmarks[i], landmarks[i+1]
        
        # Skip if landmark is not visible (zero coordinates)
        if x == 0 and y == 0:
            polar_landmarks.extend([0, 0])
            continue
        
        # Convert to polar coordinates
        r, theta = cart2polar(x, y, center_x, center_y)
        
        # Normalize radius if requested
        if normalize_radius:
            r /= diagonal
        
        polar_landmarks.extend([r, theta])
    
    return polar_landmarks


def polar_to_landmarks(polar_landmarks, img_width, img_height, normalize_radius=True):
    """
    Convert landmarks from polar to Cartesian coordinates.
    
    Args:
        polar_landmarks: Landmarks in polar coordinates [r1, theta1, r2, theta2, ..., rn, thetan]
        img_width: Image width
        img_height: Image height
        normalize_radius: Whether radius was normalized by image diagonal
        
    Returns:
        landmarks: Landmarks in format [x1, y1, x2, y2, ..., xn, yn]
    """
    # Image center as reference point
    center_x = img_width / 2
    center_y = img_height / 2
    
    # Diagonal for denormalization
    diagonal = np.sqrt(img_width**2 + img_height**2) if normalize_radius else 1.0
    
    landmarks = []
    for i in range(0, len(polar_landmarks), 2):
        r, theta = polar_landmarks[i], polar_landmarks[i+1]
        
        # Skip if landmark is not visible (zero coordinates)
        if r == 0 and theta == 0:
            landmarks.extend([0, 0])
            continue
        
        # Denormalize radius if it was normalized
        if normalize_radius:
            r *= diagonal
        
        # Convert to Cartesian coordinates
        x, y = polar2cart(r, theta, center_x, center_y)
        landmarks.extend([x, y])
    
    return landmarks


def parse_widerface_annotation(annotation_file):
    """
    Parse WiderFace annotation file.
    
    Args:
        annotation_file: Path to annotation file
        
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
            if line_idx < len(lines):
                face_count = int(lines[line_idx].strip())
                line_idx += 1
            
            # Parse face annotations
            for _ in range(face_count):
                if line_idx < len(lines):
                    face_anno = lines[line_idx].strip().split()
                    line_idx += 1
                    
                    # Parse bbox (x, y, w, h)
                    if len(face_anno) >= 4:
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
        else:
            line_idx += 1
    
    return annotations


def parse_widerface_landmarks(landmarks_file):
    """
    Parse WiderFace landmarks file.
    
    Args:
        landmarks_file: Path to landmarks file
        
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
            if line_idx < len(lines):
                face_count = int(lines[line_idx].strip())
                line_idx += 1
            
            # Parse landmark annotations
            for _ in range(face_count):
                if line_idx < len(lines):
                    landmark_anno = lines[line_idx].strip().split()
                    line_idx += 1
                    
                    # Parse landmarks (x1, y1, x2, y2, ..., x5, y5)
                    if len(landmark_anno) >= 10:
                        landmark_coords = [float(x) for x in landmark_anno[:10]]
                        landmarks[current_image].append(landmark_coords)
        else:
            line_idx += 1
    
    return landmarks


class WiderFaceDataset(Dataset):
    """
    WiderFace dataset.
    """
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms=None,
        is_train=True
    ):
        """
        Initialize the WiderFace dataset.
        
        Args:
            img_folder: Path to the image folder
            ann_file: Path to the annotation file
            transforms: Image transformations
            is_train: Whether this is a training dataset
        """
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.is_train = is_train
        
        # Parse annotations
        self.annotations = parse_widerface_annotation(ann_file)
        
        # Get list of image paths
        self.img_paths = list(self.annotations.keys())
        
        # Filter out images with no faces
        self.img_paths = [img_path for img_path in self.img_paths if self.annotations[img_path]]
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Load image
        img = Image.open(os.path.join(self.img_folder, img_path)).convert('RGB')
        
        # Get annotations for this image
        annotations = self.annotations[img_path]
        
        # Extract bounding boxes and labels
        boxes = [anno['bbox'] for anno in annotations]
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # All faces have label 1
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': idx,
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            'orig_size': torch.as_tensor([img.height, img.width]),
            'size': torch.as_tensor([img.height, img.width])
        }
        
        # Apply transformations
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


class WiderFaceLandmarksDataset(Dataset):
    """
    WiderFace dataset with 5 landmarks support.
    """
    def __init__(
        self,
        img_folder,
        ann_file,
        landmarks_file=None,
        transforms=None,
        is_train=True,
        use_polar=False,
        filter_invalid=True,
        min_face_size=8,
        cache_mode=False,
        exclude_images: List[str] = None,
        image_prefix: str = ""
    ):
        """
        Initialize the WiderFace dataset with landmarks.
        
        Args:
            img_folder: Path to the image folder
            ann_file: Path to the annotation file
            landmarks_file: Path to the landmarks file (optional)
            transforms: Image transformations
            is_train: Whether this is a training dataset
            use_polar: Whether to use polar coordinate representation
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
        self.landmarks = {}
        if landmarks_file is not None:
            self.landmarks = parse_widerface_landmarks(landmarks_file)
        
        # Merge landmarks into annotations if available
        if self.landmarks:
            for img_path, faces in self.annotations.items():
                if img_path in self.landmarks:
                    img_landmarks = self.landmarks[img_path]
                    for i, face in enumerate(faces):
                        if i < len(img_landmarks):
                            face['landmarks'] = img_landmarks[i]
        
        # Get list of image paths
        self.img_paths = list(self.annotations.keys())
        
        # Filter out excluded images
        if self.exclude_images:
            self.img_paths = [img_path for img_path in self.img_paths if img_path not in self.exclude_images]
        
        # Filter out images with no faces or invalid faces
        valid_img_paths = []
        for img_path in self.img_paths:
            faces = self.annotations[img_path]
            valid_faces = []
            
            for face in faces:
                # Filter out invalid faces
                if self.filter_invalid and face.get('invalid', 0) == 1:
                    continue
                
                # Get bbox
                bbox = face.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                # Filter out small faces
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < self.min_face_size or height < self.min_face_size:
                    continue
                
                valid_faces.append(face)
            
            if valid_faces:
                self.annotations[img_path] = valid_faces
                valid_img_paths.append(img_path)
        
        self.img_paths = valid_img_paths
        
        # Initialize cache
        self.cache = {}
    
    def __len__(self):
        return len(self.img_paths)
    
    def get_image(self, img_path):
        """
        Load image from disk or cache.
        
        Args:
            img_path: Path to the image
            
        Returns:
            img: PIL Image
        """
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
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Load image
        img = self.get_image(img_path)
        w, h = img.size
        
        # Get annotations for this image
        annotations = self.annotations[img_path]
        
        # Extract bounding boxes, landmarks, and labels
        boxes = []
        landmarks = []
        labels = []
        
        for face in annotations:
            # Get bbox
            bbox = face.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            # Get landmarks if available
            face_landmarks = face.get('landmarks', [])
            if len(face_landmarks) != 10:  # 5 landmarks, each with x and y
                face_landmarks = [0.0] * 10  # Default to all zeros
            
            # Convert to polar coordinates if needed
            if self.use_polar:
                bbox = bbox_to_polar(bbox, w, h)
                if any(face_landmarks):  # Only convert if landmarks are present
                    face_landmarks = landmarks_to_polar(face_landmarks, w, h)
            
            boxes.append(bbox)
            landmarks.append(face_landmarks)
            labels.append(1)  # All faces have label 1
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        landmarks = torch.as_tensor(landmarks, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'landmarks': landmarks,
            'labels': labels,
            'image_id': idx,
            'img_path': img_path,
            'orig_size': torch.as_tensor([h, w]),
            'size': torch.as_tensor([h, w]),
            'use_polar': self.use_polar
        }
        
        # Apply transformations
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


def collate_fn(batch):
    """
    Collate function for data loader.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        images: Tensor of images
        targets: List of targets
    """
    # Unzip list of tuples
    images, targets = list(zip(*batch))

    # NOTE:
    # Images can have different spatial resolutions, so we return them
    # as a list instead of stacking into a single tensor.  The training
    # loop or a downstream data–prefetcher is expected to handle
    # per-sample tensor operations (e.g. padding, batching on-device).
    #
    # This is the standard collate strategy for object-detection tasks.
    return list(images), list(targets)


def build_widerface(image_set, args):
    """
    Build WiderFace dataset with landmarks.
    
    Args:
        image_set: Dataset split ('train' or 'val')
        args: Arguments
        
    Returns:
        dataset: WiderFace dataset with landmarks
        num_classes: Number of classes (2 for face detection)
    """
    """
    Helper **now handles both**:
      • an argparse‐like object with attributes
      • a (possibly nested) dict coming from YAML config
    """
    def _get_val(key, default=None):
        # Dict style
        if isinstance(args, dict):
            if key in args:                      # flat
                return args.get(key, default)
            # look into typical sub-dicts
            for sub in ("dataset", "data"):
                if sub in args and key in args[sub]:
                    return args[sub].get(key, default)
            return default
        # Namespace / object style
        return getattr(args, key, default)

    # ------------------------------------------------------------------
    # Required root path
    # ------------------------------------------------------------------
    root_path = _get_val("widerface_path")
    if root_path is None:
        raise AttributeError(
            "`widerface_path` not found in provided config/args."
        )
    root = Path(root_path)
    assert root.exists(), f"WiderFace path {root} does not exist"
    
    # ---------------------------
    #   Paths & Prefix handling
    # ---------------------------
    # Images live under root / <image_prefix> / <json-path>
    # Example: json has '0--Parade/xxx.jpg'
    #   widerface_path = '/data/widerface'
    #   image_prefix   = 'WIDER_train/images'
    #   final path     = /data/widerface/WIDER_train/images/0--Parade/xxx.jpg
    image_prefix = _get_val("image_prefix", "")

    # Set annotation file paths based on split
    if image_set == 'train':
        img_folder = root  # base path; prefix will be applied later
        # Use the JSON annotation produced by the user
        ann_file = root / 'annotations' / 'train_wider_face.json'
        landmarks_file = None  # landmarks embedded in JSON
    elif image_set == 'val':
        img_folder = root
        ann_file = root / 'annotations' / 'val_wider_face.json'
        landmarks_file = None
    else:
        raise ValueError(f"Unknown image_set: {image_set}")
    
    # Create transformations
    # Common basic transforms: convert PIL → tensor and normalise.
    basic_transforms = [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ]

    if image_set == 'train':
        transforms = Compose([
            # Add your training augmentations here (e.g. flips, crops …),
            *basic_transforms,
        ])
    else:
        transforms = Compose([
            # Validation / test require only deterministic transforms
            *basic_transforms,
        ])
    
    # Create dataset
    dataset = WiderFaceLandmarksDataset(
        img_folder=str(img_folder),
        ann_file=str(ann_file),
        landmarks_file=str(landmarks_file) if (landmarks_file and landmarks_file.exists()) else None,
        transforms=transforms,
        is_train=(image_set == 'train'),
        use_polar=_get_val("use_polar", False),
        filter_invalid=_get_val("filter_invalid", True),
        min_face_size=_get_val("min_face_size", 8),
        cache_mode=_get_val("cache_mode", False),
        image_prefix=image_prefix,
    )
    
    # Face detection has 2 classes: background (0) and face (1)
    num_classes = 2
    
    return dataset, num_classes
