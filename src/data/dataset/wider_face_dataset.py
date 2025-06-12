"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

WIDER Face Dataset with face detection and landmark localization support s
"""

import os
import json
import torch
import torchvision
from PIL import Image
from typing import Optional, Callable, List, Dict, Any
import numpy as np

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class WiderFaceDetection(DetDataset):
    """WIDER Face dataset for face detection and landmark localization"""
    
    __inject__ = ['transforms']
    
    def __init__(
        self,
        root: str,
        ann_file: str,
        image_set: str = 'train',
        transforms: Optional[Callable] = None,
        return_landmarks: bool = True,
        min_face_size: int = 16,
        use_synthetic_landmarks: bool = True
    ):
        """
        Args:
            root: Dataset root directory
            ann_file: Annotation file path (JSON format)
            image_set: 'train' or 'val'
            transforms: Data transforms
            return_landmarks: Whether to return landmark annotations
            min_face_size: Minimum face size to include
            use_synthetic_landmarks: Use synthetic landmarks if real ones not available
        """
        self.root = root
        self.ann_file = ann_file
        self.image_set = image_set
        self.transforms = transforms
        self.return_landmarks = return_landmarks
        self.min_face_size = min_face_size
        self.use_synthetic_landmarks = use_synthetic_landmarks
        
        # Load annotations
        self.annotations = self._load_annotations()
        self.image_ids = list(self.annotations.keys())
        
        # WIDER Face categories
        self.categories = [{"id": 1, "name": "face"}]
        
    def _load_annotations(self) -> Dict:
        """Load and parse annotation file"""
        if self.ann_file.endswith('.json'):
            with open(self.ann_file, 'r') as f:
                annotations = json.load(f)
        else:
            # Parse WIDER Face format annotation file
            annotations = self._parse_wider_face_annotations()
            
        # Filter by minimum face size
        filtered_annotations = {}
        for image_path, data in annotations.items():
            valid_faces = []
            for face in data['faces']:
                bbox = face['bbox']
                if bbox[2] >= self.min_face_size and bbox[3] >= self.min_face_size:
                    valid_faces.append(face)
            
            if valid_faces:
                data_copy = data.copy()
                data_copy['faces'] = valid_faces
                filtered_annotations[image_path] = data_copy
                
        return filtered_annotations
    
    def _parse_wider_face_annotations(self) -> Dict:
        """Parse original WIDER Face annotation format"""
        annotations = {}
        
        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Image path
            image_path = lines[i].strip()
            i += 1
            
            # Check if image exists
            full_image_path = os.path.join(self.root, image_path)
            if not os.path.exists(full_image_path):
                # Skip this image
                if i < len(lines):
                    num_faces = int(lines[i].strip())
                    i += 1 + max(num_faces, 1)
                continue
            
            # Number of faces
            num_faces = int(lines[i].strip())
            i += 1
            
            faces = []
            
            # Process each face
            for j in range(max(num_faces, 1)):
                if i >= len(lines):
                    break
                    
                line = lines[i].strip()
                i += 1
                
                if num_faces == 0:
                    break
                
                # Parse face annotation
                parts = line.split()
                if len(parts) >= 4:
                    x, y, w, h = map(int, parts[:4])
                    
                    # Additional attributes
                    blur = int(parts[4]) if len(parts) > 4 else 0
                    expression = int(parts[5]) if len(parts) > 5 else 0
                    illumination = int(parts[6]) if len(parts) > 6 else 0
                    invalid = int(parts[7]) if len(parts) > 7 else 0
                    occlusion = int(parts[8]) if len(parts) > 8 else 0
                    pose = int(parts[9]) if len(parts) > 9 else 0
                    
                    # Only include valid faces
                    if invalid == 0:
                        face_data = {
                            'bbox': [x, y, w, h],
                            'blur': blur,
                            'expression': expression,
                            'illumination': illumination,
                            'occlusion': occlusion,
                            'pose': pose,
                            'landmarks': None
                        }
                        faces.append(face_data)
            
            if faces:
                # Get image dimensions
                try:
                    with Image.open(full_image_path) as img:
                        img_width, img_height = img.size
                    
                    annotations[image_path] = {
                        'image_width': img_width,
                        'image_height': img_height,
                        'faces': faces
                    }
                except Exception as e:
                    print(f"Error loading image {full_image_path}: {e}")
                    continue
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int):
        image, target = self.load_item(idx)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)
        return image, target
    
    def load_item(self, idx: int):
        """Load image and target at given index"""
        image_path = self.image_ids[idx]
        image_data = self.annotations[image_path]
        
        # Load image
        full_image_path = os.path.join(self.root, image_path)
        image = Image.open(full_image_path).convert('RGB')
        
        # Prepare target
        target = {
            'image_id': torch.tensor([idx]),
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': [],
            'landmarks': [],
            'face_attributes': []
        }
        
        for face in image_data['faces']:
            bbox = face['bbox']
            x, y, w, h = bbox
            
            # Convert to xyxy format
            target['boxes'].append([x, y, x + w, y + h])
            target['labels'].append(0)  # Face class
            target['area'].append(w * h)
            target['iscrowd'].append(0)
            
            # Handle landmarks
            if self.return_landmarks:
                if face.get('landmarks') is not None:
                    target['landmarks'].append(face['landmarks'])
                elif self.use_synthetic_landmarks:
                    # Generate synthetic 5-point landmarks
                    landmarks = self._generate_synthetic_landmarks(bbox)
                    target['landmarks'].append(landmarks)
                else:
                    target['landmarks'].append([0] * 10)  # 5 points * 2 coordinates
            
            # Face attributes for auxiliary tasks
            target['face_attributes'].append({
                'blur': face.get('blur', 0),
                'expression': face.get('expression', 0),
                'illumination': face.get('illumination', 0),
                'occlusion': face.get('occlusion', 0),
                'pose': face.get('pose', 0)
            })
        
        # Convert to tensors
        w, h = image.size
        if target['boxes']:
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
            target['boxes'] = convert_to_tv_tensor(
                target['boxes'], 'boxes', box_format='xyxy', spatial_size=[h, w])
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        target['area'] = torch.tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.int64)
        target['orig_size'] = torch.tensor([w, h])
        
        if self.return_landmarks and target['landmarks']:
            target['landmarks'] = torch.tensor(target['landmarks'], dtype=torch.float32)
        else:
            target['landmarks'] = torch.zeros((len(target['labels']), 10), dtype=torch.float32)
        
        return image, target
    
    def _generate_synthetic_landmarks(self, bbox: List[int]) -> List[float]:
        """Generate synthetic 5-point landmarks for a face bbox"""
        x, y, w, h = bbox
        
        # Add some randomness for training diversity
        noise_scale = 0.05
        
        landmarks = []
        
        # Left eye
        left_eye_x = x + w * (0.3 + np.random.normal(0, noise_scale))
        left_eye_y = y + h * (0.35 + np.random.normal(0, noise_scale))
        landmarks.extend([left_eye_x, left_eye_y])
        
        # Right eye
        right_eye_x = x + w * (0.7 + np.random.normal(0, noise_scale))
        right_eye_y = y + h * (0.35 + np.random.normal(0, noise_scale))
        landmarks.extend([right_eye_x, right_eye_y])
        
        # Nose
        nose_x = x + w * (0.5 + np.random.normal(0, noise_scale))
        nose_y = y + h * (0.55 + np.random.normal(0, noise_scale))
        landmarks.extend([nose_x, nose_y])
        
        # Left mouth corner
        left_mouth_x = x + w * (0.35 + np.random.normal(0, noise_scale))
        left_mouth_y = y + h * (0.75 + np.random.normal(0, noise_scale))
        landmarks.extend([left_mouth_x, left_mouth_y])
        
        # Right mouth corner
        right_mouth_x = x + w * (0.65 + np.random.normal(0, noise_scale))
        right_mouth_y = y + h * (0.75 + np.random.normal(0, noise_scale))
        landmarks.extend([right_mouth_x, right_mouth_y])
        
        return landmarks
    
    @property
    def category2name(self) -> Dict:
        return {1: 'face'}
    
    @property
    def category2label(self) -> Dict:
        return {1: 0}
    
    @property
    def label2category(self) -> Dict:
        return {0: 1}