"""
Data transformations for Polar-RTDETRv2.

This module provides transformations for the WiderFace dataset with 5 landmarks.
It includes:
- Basic image transformations
- Coordinate transformations
- Landmark-aware augmentations
- Normalization and tensor conversion
"""

import random
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding box format from [x1, y1, x2, y2] to [cx, cy, w, h].
    
    Args:
        x: Bounding box tensor in [x1, y1, x2, y2] format
        
    Returns:
        Bounding box tensor in [cx, cy, w, h] format
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box format from [cx, cy, w, h] to [x1, y1, x2, y2].
    
    Args:
        x: Bounding box tensor in [cx, cy, w, h] format
        
    Returns:
        Bounding box tensor in [x1, y1, x2, y2] format
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def convert_landmarks_to_tensor(landmarks, device=None):
    """
    Convert landmarks to tensor.
    
    Args:
        landmarks: Landmarks in list or numpy array format
        device: Device to put tensor on
        
    Returns:
        landmarks_tensor: Landmarks as tensor
    """
    if isinstance(landmarks, torch.Tensor):
        return landmarks.to(device) if device else landmarks
    
    return torch.tensor(landmarks, dtype=torch.float32, device=device)


class Compose:
    """
    Compose multiple transforms together.
    """
    def __init__(self, transforms):
        """
        Initialize Compose.
        
        Args:
            transforms: List of transforms
        """
        self.transforms = transforms

    def __call__(self, image, target):
        """
        Apply transforms to image and target.
        
        Args:
            image: PIL Image
            target: Target dictionary
            
        Returns:
            image: Transformed image
            target: Transformed target
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """
    Convert PIL Image to tensor and normalize.
    """
    def __call__(self, image, target):
        """
        Convert image to tensor and leave target unchanged.
        
        Args:
            image: PIL Image
            target: Target dictionary
            
        Returns:
            image: Tensor image
            target: Target dictionary
        """
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """
    Normalize tensor image with mean and standard deviation.
    """
    def __init__(self, mean, std):
        """
        Initialize Normalize.
        
        Args:
            mean: Mean for each channel
            std: Standard deviation for each channel
        """
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        """
        Normalize image and leave target unchanged.
        
        Args:
            image: Tensor image
            target: Target dictionary
            
        Returns:
            image: Normalized tensor image
            target: Target dictionary
        """
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip:
    """
    Randomly flip image horizontally.
    """
    def __init__(self, prob=0.5):
        """
        Initialize RandomHorizontalFlip.
        
        Args:
            prob: Probability of flipping
        """
        self.prob = prob

    def __call__(self, image, target):
        """
        Flip image and target horizontally with probability prob.
        
        Args:
            image: PIL Image or Tensor
            target: Target dictionary
            
        Returns:
            image: Flipped image
            target: Flipped target
        """
        if random.random() < self.prob:
            width, height = image.size if isinstance(image, Image.Image) else (image.shape[2], image.shape[1])
            image = F.hflip(image)
            
            # Flip bounding boxes
            if 'boxes' in target:
                boxes = target['boxes']
                boxes = boxes.clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes
            
            # Flip landmarks
            if 'landmarks' in target:
                landmarks = target['landmarks']
                landmarks = landmarks.clone()
                
                # Check if using polar coordinates
                use_polar = target.get('use_polar', False)
                
                if use_polar:
                    # For polar coordinates, we need to adjust the angles
                    for i in range(0, landmarks.shape[1], 2):
                        # Only flip angle (odd indices), not radius (even indices)
                        landmarks[:, i+1] = -landmarks[:, i+1]
                        
                    # Swap left and right landmarks (0<->1, 3<->4)
                    # Right eye (0) <-> Left eye (1)
                    landmarks[:, [0, 1, 2, 3]] = landmarks[:, [2, 3, 0, 1]]
                    # Right mouth (3) <-> Left mouth (4)
                    landmarks[:, [6, 7, 8, 9]] = landmarks[:, [8, 9, 6, 7]]
                else:
                    # For Cartesian coordinates, flip x-coordinates
                    for i in range(0, landmarks.shape[1], 2):
                        landmarks[:, i] = width - landmarks[:, i]
                    
                    # Swap left and right landmarks
                    # Right eye (0,1) <-> Left eye (2,3)
                    landmarks[:, [0, 1, 2, 3]] = landmarks[:, [2, 3, 0, 1]]
                    # Right mouth (6,7) <-> Left mouth (8,9)
                    landmarks[:, [6, 7, 8, 9]] = landmarks[:, [8, 9, 6, 7]]
                
                target['landmarks'] = landmarks
        
        return image, target


class RandomResize:
    """
    Randomly resize image.
    """
    def __init__(self, min_size, max_size=None):
        """
        Initialize RandomResize.
        
        Args:
            min_size: Minimum size (can be a list for random selection)
            max_size: Maximum size
        """
        self.min_size = min_size if isinstance(min_size, (list, tuple)) else [min_size]
        self.max_size = max_size

    def __call__(self, image, target):
        """
        Resize image and target.
        
        Args:
            image: PIL Image
            target: Target dictionary
            
        Returns:
            image: Resized image
            target: Resized target
        """
        size = random.choice(self.min_size)
        return resize(image, target, size, self.max_size)


class RandomSelect:
    """
    Randomly select one of the transforms with probability p.
    """
    def __init__(self, transform1, transform2, p=0.5):
        """
        Initialize RandomSelect.
        
        Args:
            transform1: First transform
            transform2: Second transform
            p: Probability of selecting transform1
        """
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(self, image, target):
        """
        Apply either transform1 or transform2.
        
        Args:
            image: PIL Image
            target: Target dictionary
            
        Returns:
            image: Transformed image
            target: Transformed target
        """
        if random.random() < self.p:
            return self.transform1(image, target)
        return self.transform2(image, target)


class RandomSizeCrop:
    """
    Randomly crop image with size between min_size and max_size.
    """
    def __init__(self, min_size, max_size, respect_boxes=True):
        """
        Initialize RandomSizeCrop.
        
        Args:
            min_size: Minimum crop size
            max_size: Maximum crop size
            respect_boxes: Whether to ensure boxes are not cropped too much
        """
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes

    def __call__(self, image, target):
        """
        Crop image and adjust target.
        
        Args:
            image: PIL Image
            target: Target dictionary
            
        Returns:
            image: Cropped image
            target: Adjusted target
        """
        width, height = image.size
        
        # Determine crop size
        w = random.randint(self.min_size, min(width, self.max_size))
        h = random.randint(self.min_size, min(height, self.max_size))
        
        # If we need to respect boxes, ensure crop doesn't cut too many boxes
        if self.respect_boxes and 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            
            # Try a few times to find a good crop
            for _ in range(10):
                left = random.randint(0, width - w)
                top = random.randint(0, height - h)
                
                # Calculate IoU between boxes and crop
                crop_box = torch.tensor([[left, top, left + w, top + h]])
                
                # Calculate intersection
                x1 = torch.max(boxes[:, 0], crop_box[0, 0])
                y1 = torch.max(boxes[:, 1], crop_box[0, 1])
                x2 = torch.min(boxes[:, 2], crop_box[0, 2])
                y2 = torch.min(boxes[:, 3], crop_box[0, 3])
                
                # Calculate area of intersection
                w_inter = torch.clamp(x2 - x1, min=0)
                h_inter = torch.clamp(y2 - y1, min=0)
                area_inter = w_inter * h_inter
                
                # Calculate area of boxes
                area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                
                # Calculate IoU
                iou = area_inter / area_boxes
                
                # If at least 60% of all boxes are mostly inside the crop, use this crop
                if (iou > 0.6).sum() > len(boxes) * 0.6:
                    break
            else:
                # If no good crop found, just use random crop
                left = random.randint(0, width - w)
                top = random.randint(0, height - h)
        else:
            # Random crop
            left = random.randint(0, width - w)
            top = random.randint(0, height - h)
        
        # Crop image
        image = F.crop(image, top, left, h, w)
        
        # Adjust target
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([left, top, left, top])
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            
            # Keep only valid boxes
            keep = (cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :]).all(dim=1)
            
            if keep.sum().item() < len(boxes):
                # Some boxes were removed, adjust all target elements
                for k in target.keys():
                    if k == 'boxes':
                        target[k] = cropped_boxes.reshape(-1, 4)[keep]
                    elif k in ['labels', 'area', 'iscrowd', 'landmarks']:
                        target[k] = target[k][keep]
            else:
                target['boxes'] = cropped_boxes.reshape(-1, 4)
            
            # Adjust landmarks
            if 'landmarks' in target and len(target['landmarks']) > 0:
                landmarks = target['landmarks']
                
                # Check if using polar coordinates
                use_polar = target.get('use_polar', False)
                
                if use_polar:
                    # For polar coordinates, we need to adjust the center point
                    # This is complex and requires recalculating all polar coordinates
                    # For simplicity, we'll just mark landmarks that are outside the crop as invisible
                    
                    # Convert landmarks to Cartesian for checking visibility
                    center_x, center_y = width / 2, height / 2
                    new_center_x, new_center_y = (width - left - w) / 2, (height - top - h) / 2
                    
                    # For each landmark pair (r, theta)
                    for i in range(0, landmarks.shape[1], 2):
                        r, theta = landmarks[:, i], landmarks[:, i+1]
                        
                        # Convert to Cartesian
                        x = center_x + r * torch.cos(theta)
                        y = center_y + r * torch.sin(theta)
                        
                        # Check if inside crop
                        inside = (x >= left) & (x < left + w) & (y >= top) & (y < top + h)
                        
                        # Set invisible if outside
                        landmarks[~inside, i] = 0
                        landmarks[~inside, i+1] = 0
                        
                        # Adjust coordinates for those inside
                        if inside.any():
                            # Recalculate polar coordinates with new center
                            x_new = x[inside] - left
                            y_new = y[inside] - top
                            
                            dx = x_new - new_center_x
                            dy = y_new - new_center_y
                            
                            r_new = torch.sqrt(dx**2 + dy**2)
                            theta_new = torch.atan2(dy, dx)
                            
                            landmarks[inside, i] = r_new
                            landmarks[inside, i+1] = theta_new
                else:
                    # For Cartesian coordinates, adjust x,y coordinates
                    for i in range(0, landmarks.shape[1], 2):
                        # Adjust x-coordinates
                        landmarks[:, i] = landmarks[:, i] - left
                        # Adjust y-coordinates
                        landmarks[:, i+1] = landmarks[:, i+1] - top
                        
                        # Check if landmark is inside crop
                        inside = (landmarks[:, i] >= 0) & (landmarks[:, i] < w) & \
                                (landmarks[:, i+1] >= 0) & (landmarks[:, i+1] < h)
                        
                        # Set to 0 if outside (invisible)
                        landmarks[~inside, i] = 0
                        landmarks[~inside, i+1] = 0
                
                target['landmarks'] = landmarks[keep]
        
        # Update size
        target['size'] = torch.tensor([h, w])
        
        return image, target


class LandmarkAugmentation:
    """
    Apply landmark-specific augmentations.
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, blur_prob=0.1, noise_prob=0.1):
        """
        Initialize LandmarkAugmentation.
        
        Args:
            brightness: Brightness jitter factor
            contrast: Contrast jitter factor
            saturation: Saturation jitter factor
            hue: Hue jitter factor
            blur_prob: Probability of applying blur
            noise_prob: Probability of adding noise
        """
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob

    def __call__(self, image, target):
        """
        Apply augmentations to image while preserving landmarks.
        
        Args:
            image: PIL Image
            target: Target dictionary
            
        Returns:
            image: Augmented image
            target: Target dictionary (unchanged)
        """
        # Apply color jitter
        image = self.color_jitter(image)
        
        # Apply blur with probability
        if random.random() < self.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # Apply noise with probability
        if random.random() < self.noise_prob and isinstance(image, Image.Image):
            img_np = np.array(image)
            noise = np.random.normal(0, 5, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_np)
        
        return image, target


def resize(image, target, size, max_size=None):
    """
    Resize image and target.
    
    Args:
        image: PIL Image
        target: Target dictionary
        size: Target size
        max_size: Maximum size
        
    Returns:
        image: Resized image
        target: Resized target
    """
    # Determine size
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        
        return (oh, ow)
    
    # Get target size
    size = get_size_with_aspect_ratio(image.size, size, max_size)
    
    # Resize image
    image = F.resize(image, size)
    
    # Adjust target
    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes']
        orig_size = torch.tensor([target['orig_size'][1], target['orig_size'][0]], dtype=torch.float32)
        new_size = torch.tensor([size[1], size[0]], dtype=torch.float32)
        scale_factor = new_size / orig_size
        
        # Scale boxes
        boxes = boxes * scale_factor.repeat(2)
        target['boxes'] = boxes
        
        # Scale landmarks if present
        if 'landmarks' in target and len(target['landmarks']) > 0:
            landmarks = target['landmarks']
            
            # Check if using polar coordinates
            use_polar = target.get('use_polar', False)
            
            if use_polar:
                # For polar coordinates, only scale radius (even indices)
                for i in range(0, landmarks.shape[1], 2):
                    landmarks[:, i] = landmarks[:, i] * scale_factor.mean()
            else:
                # For Cartesian coordinates, scale x,y coordinates
                for i in range(0, landmarks.shape[1], 2):
                    landmarks[:, i] = landmarks[:, i] * scale_factor[0]
                    landmarks[:, i+1] = landmarks[:, i+1] * scale_factor[1]
            
            target['landmarks'] = landmarks
    
    # Update size
    target['size'] = torch.tensor([size[0], size[1]])
    
    return image, target
