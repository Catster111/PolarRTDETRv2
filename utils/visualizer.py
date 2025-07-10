"""
Visualizer module for Polar-RTDETRv2.

This module provides visualization utilities for Polar-RTDETRv2, including:
- Visualization of detection results (boxes and landmarks)
- Plotting of training metrics
- Visualization of model predictions
- Support for both Cartesian and polar coordinate systems
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any


class Visualizer:
    """
    Visualizer for Polar-RTDETRv2.
    
    This class provides methods for visualizing detection results, training metrics,
    and model predictions. It supports both Cartesian and polar coordinate systems.
    """
    
    def __init__(
        self,
        output_dir: str = 'outputs/visualizations',
        landmark_colors: List[str] = None,
        box_color: str = 'g',
        font_path: Optional[str] = None,
        font_size: int = 12,
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            landmark_colors: Colors for each landmark (default: ['r', 'b', 'y', 'm', 'c'])
            box_color: Color for bounding boxes (default: 'g')
            font_path: Path to font file for text rendering
            font_size: Font size for text
            dpi: DPI for saved figures
            figsize: Figure size for plots
        """
        self.output_dir = output_dir
        self.landmark_colors = landmark_colors or ['r', 'b', 'y', 'm', 'c']
        self.box_color = box_color
        self.font_path = font_path
        self.font_size = font_size
        self.dpi = dpi
        self.figsize = figsize
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to load font if specified
        self.font = None
        if font_path and os.path.exists(font_path):
            try:
                self.font = ImageFont.truetype(font_path, font_size)
            except Exception:
                print(f"Warning: Failed to load font from {font_path}")
        
        # RGB values for matplotlib colors (for PIL/OpenCV)
        self.color_map = {
            'r': (255, 0, 0),
            'g': (0, 255, 0),
            'b': (0, 0, 255),
            'y': (255, 255, 0),
            'm': (255, 0, 255),
            'c': (0, 255, 255),
            'w': (255, 255, 255),
            'k': (0, 0, 0)
        }
    
    def visualize_batch(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]] = None,
        predictions: Dict[str, torch.Tensor] = None,
        max_images: int = 8,
        save_path: Optional[str] = None,
        show: bool = False,
        return_fig: bool = False
    ) -> Optional[Figure]:
        """
        Visualize a batch of images with targets and/or predictions.
        
        Args:
            images: Batch of images [B, C, H, W]
            targets: List of target dictionaries
            predictions: Dictionary of predictions
            max_images: Maximum number of images to visualize
            save_path: Path to save visualization
            show: Whether to show the visualization
            return_fig: Whether to return the figure
            
        Returns:
            fig: Figure if return_fig is True, otherwise None
        """
        # Determine number of images to visualize
        batch_size = images.shape[0]
        num_images = min(batch_size, max_images)
        
        # Create figure
        fig, axes = plt.subplots(num_images, 1, figsize=(self.figsize[0], self.figsize[1] * num_images))
        if num_images == 1:
            axes = [axes]
        
        # Visualize each image
        for i in range(num_images):
            # Get image
            img = images[i].detach().cpu()
            
            # Convert to numpy and denormalize if needed
            if img.shape[0] == 3:  # CHW format
                img = img.permute(1, 2, 0).numpy()
            else:  # HWC format
                img = img.numpy()
            
            # Denormalize if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            # Get target and prediction for this image
            target = targets[i] if targets is not None else None
            pred = {k: v[i:i+1] for k, v in predictions.items()} if predictions is not None else None
            
            # Plot image
            axes[i].imshow(img)
            
            # Draw target boxes and landmarks
            if target is not None:
                self._draw_boxes_and_landmarks(
                    axes[i],
                    target.get('boxes', None),
                    target.get('landmarks', None),
                    target.get('labels', None),
                    target.get('scores', None),
                    target.get('use_polar', False),
                    img.shape[:2],  # (H, W)
                    alpha=0.3,
                    linewidth=2,
                    markersize=8,
                    label_prefix='GT'
                )
            
            # Draw prediction boxes and landmarks
            if pred is not None:
                self._draw_boxes_and_landmarks(
                    axes[i],
                    pred.get('pred_boxes', None),
                    pred.get('pred_landmarks', None),
                    pred.get('pred_logits', None),
                    pred.get('pred_scores', None),
                    predictions.get('use_polar', False),
                    img.shape[:2],  # (H, W)
                    alpha=1.0,
                    linewidth=1,
                    markersize=6,
                    label_prefix='Pred'
                )
            
            # Set title
            axes[i].set_title(f"Image {i}")
            axes[i].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.show()
        
        # Return figure if requested
        if return_fig:
            return fig
        
        # Close figure if not showing or returning
        if not show and not return_fig:
            plt.close(fig)
        
        return None
    
    def visualize_predictions(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        boxes: Optional[torch.Tensor] = None,
        landmarks: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
        use_polar: bool = False,
        class_names: Optional[List[str]] = None,
        score_threshold: float = 0.5,
        save_path: Optional[str] = None,
        show: bool = False,
        return_image: bool = False
    ) -> Optional[np.ndarray]:
        """
        Visualize predictions on a single image.
        
        Args:
            image: Image to visualize (tensor, numpy array, or PIL image)
            boxes: Predicted boxes [N, 4]
            landmarks: Predicted landmarks [N, num_landmarks*2]
            labels: Predicted labels [N]
            scores: Predicted scores [N]
            use_polar: Whether boxes and landmarks are in polar coordinates
            class_names: List of class names
            score_threshold: Threshold for filtering predictions by score
            save_path: Path to save visualization
            show: Whether to show the visualization
            return_image: Whether to return the image
            
        Returns:
            img_np: Numpy array of image if return_image is True, otherwise None
        """
        # Convert image to numpy array
        img_np = self._to_numpy(image)
        
        # Create PIL image for drawing
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        
        # Get image dimensions
        height, width = img_np.shape[:2]
        
        # Filter predictions by score if provided
        if scores is not None and boxes is not None:
            keep = scores > score_threshold
            boxes = boxes[keep]
            if landmarks is not None:
                landmarks = landmarks[keep]
            if labels is not None:
                labels = labels[keep]
            scores = scores[keep]
        
        # Draw boxes
        if boxes is not None:
            boxes_np = boxes.detach().cpu().numpy()
            
            # Convert from polar to Cartesian if needed
            if use_polar:
                boxes_np = self._polar_to_cartesian_boxes(boxes_np, (height, width))
            
            # Draw each box
            for i, box in enumerate(boxes_np):
                # Get box coordinates
                x1, y1, x2, y2 = box
                
                # Scale to image size if normalized
                if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                    x1 = int(x1 * width)
                    y1 = int(y1 * height)
                    x2 = int(x2 * width)
                    y2 = int(y2 * height)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=self.color_map[self.box_color], width=2)
                
                # Draw label and score if available
                label_text = ""
                if labels is not None:
                    label = labels[i].item()
                    if class_names is not None and label < len(class_names):
                        label_text = class_names[label]
                    else:
                        label_text = f"Class {label}"
                
                if scores is not None:
                    score = scores[i].item()
                    label_text += f" {score:.2f}"
                
                if label_text:
                    # Draw text background
                    text_w, text_h = draw.textbbox((0, 0), label_text, font=self.font)[2:]
                    draw.rectangle([x1, y1 - text_h - 4, x1 + text_w, y1], fill=self.color_map[self.box_color])
                    
                    # Draw text
                    draw.text((x1, y1 - text_h - 2), label_text, fill=(255, 255, 255), font=self.font)
        
        # Draw landmarks
        if landmarks is not None:
            landmarks_np = landmarks.detach().cpu().numpy()
            
            # Convert from polar to Cartesian if needed
            if use_polar:
                landmarks_np = self._polar_to_cartesian_landmarks(landmarks_np, (height, width))
            
            # Draw each set of landmarks
            for i, lm in enumerate(landmarks_np):
                # Draw each landmark
                for j in range(len(self.landmark_colors)):
                    # Get landmark coordinates
                    x, y = lm[j*2], lm[j*2 + 1]
                    
                    # Skip if landmark is not visible (zero coordinates)
                    if x == 0 and y == 0:
                        continue
                    
                    # Scale to image size if normalized
                    if x <= 1.0 and y <= 1.0:
                        x = int(x * width)
                        y = int(y * height)
                    else:
                        x, y = int(x), int(y)
                    
                    # Draw circle
                    color = self.color_map[self.landmark_colors[j % len(self.landmark_colors)]]
                    radius = 3
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        
        # Convert back to numpy
        img_np = np.array(img_pil)
        
        # Save image if requested
        if save_path:
            img_pil.save(save_path)
        
        # Show image if requested
        if show:
            plt.figure(figsize=self.figsize)
            plt.imshow(img_np)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Return image if requested
        if return_image:
            return img_np
        
        return None
    
    def plot_metrics(
        self,
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = False,
        return_fig: bool = False,
        title: str = 'Training Metrics',
        xlabel: str = 'Epoch',
        smooth: bool = True,
        window_size: int = 10
    ) -> Optional[Figure]:
        """
        Plot training metrics.
        
        Args:
            metrics: Dictionary of metrics (key: metric name, value: list of values)
            save_path: Path to save plot
            show: Whether to show the plot
            return_fig: Whether to return the figure
            title: Plot title
            xlabel: X-axis label
            smooth: Whether to smooth the curves
            window_size: Window size for smoothing
            
        Returns:
            fig: Figure if return_fig is True, otherwise None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot each metric
        for name, values in metrics.items():
            # Skip empty values
            if not values:
                continue
            
            # Create x-axis values
            x = list(range(1, len(values) + 1))
            
            # Smooth values if requested
            if smooth and len(values) > window_size:
                values_smooth = []
                for i in range(len(values)):
                    window_start = max(0, i - window_size // 2)
                    window_end = min(len(values), i + window_size // 2 + 1)
                    values_smooth.append(np.mean(values[window_start:window_end]))
                ax.plot(x, values_smooth, label=name)
            else:
                ax.plot(x, values, label=name)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Value')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.show()
        
        # Return figure if requested
        if return_fig:
            return fig
        
        # Close figure if not showing or returning
        if not show and not return_fig:
            plt.close(fig)
        
        return None
    
    def _draw_boxes_and_landmarks(
        self,
        ax,
        boxes,
        landmarks,
        labels,
        scores,
        use_polar,
        image_size,
        alpha=1.0,
        linewidth=2,
        markersize=8,
        label_prefix=""
    ):
        """
        Draw boxes and landmarks on a matplotlib axis.
        
        Args:
            ax: Matplotlib axis
            boxes: Boxes tensor [N, 4]
            landmarks: Landmarks tensor [N, num_landmarks*2]
            labels: Labels tensor [N]
            scores: Scores tensor [N]
            use_polar: Whether boxes and landmarks are in polar coordinates
            image_size: Image size (height, width)
            alpha: Alpha value for boxes
            linewidth: Line width for boxes
            markersize: Marker size for landmarks
            label_prefix: Prefix for labels
        """
        if boxes is None:
            return
        
        # Convert tensors to numpy arrays
        boxes_np = boxes.detach().cpu().numpy()
        
        # Convert from polar to Cartesian if needed
        if use_polar:
            boxes_np = self._polar_to_cartesian_boxes(boxes_np, image_size)
        
        # Draw each box
        for i, box in enumerate(boxes_np):
            # Get box coordinates
            x1, y1, x2, y2 = box
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=linewidth,
                edgecolor=self.box_color,
                facecolor='none',
                alpha=alpha
            )
            
            # Add rectangle to axis
            ax.add_patch(rect)
            
            # Add label and score if available
            label_text = label_prefix
            if labels is not None:
                label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
                label_text += f" {label}"
            
            if scores is not None:
                score = scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i]
                label_text += f" {score:.2f}"
            
            if label_text and label_text != label_prefix:
                ax.text(
                    x1, y1 - 5,
                    label_text,
                    bbox=dict(facecolor=self.box_color, alpha=0.5),
                    fontsize=8,
                    color='white'
                )
        
        # Draw landmarks if available
        if landmarks is not None:
            landmarks_np = landmarks.detach().cpu().numpy()
            
            # Convert from polar to Cartesian if needed
            if use_polar:
                landmarks_np = self._polar_to_cartesian_landmarks(landmarks_np, image_size)
            
            # Draw each set of landmarks
            for i, lm in enumerate(landmarks_np):
                # Draw each landmark
                for j in range(len(self.landmark_colors)):
                    # Get landmark coordinates
                    x, y = lm[j*2], lm[j*2 + 1]
                    
                    # Skip if landmark is not visible (zero coordinates)
                    if x == 0 and y == 0:
                        continue
                    
                    # Draw point
                    ax.plot(
                        x, y,
                        'o',
                        color=self.landmark_colors[j % len(self.landmark_colors)],
                        markersize=markersize,
                        alpha=alpha
                    )
    
    def _to_numpy(self, image):
        """
        Convert image to numpy array.
        
        Args:
            image: Image (tensor, numpy array, or PIL image)
            
        Returns:
            img_np: Numpy array of image
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy
            img = image.detach().cpu()
            
            # Handle different tensor formats
            if img.dim() == 4 and img.shape[0] == 1:  # [1, C, H, W]
                img = img.squeeze(0)
            
            if img.dim() == 3 and img.shape[0] == 3:  # [C, H, W]
                img = img.permute(1, 2, 0)
            
            img_np = img.numpy()
            
            # Denormalize if needed
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
        elif isinstance(image, np.ndarray):
            img_np = image.copy()
        elif isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return img_np
    
    def _polar_to_cartesian_boxes(self, boxes, image_size):
        """
        Convert boxes from polar [r1, theta1, r2, theta2] to Cartesian [x1, y1, x2, y2] coordinates.
        
        Args:
            boxes: Boxes in polar coordinates [N, 4]
            image_size: Image size (height, width)
            
        Returns:
            cart_boxes: Boxes in Cartesian coordinates [N, 4]
        """
        height, width = image_size
        center_x, center_y = width / 2, height / 2
        diag = np.sqrt(width**2 + height**2)
        
        # Initialize output array
        cart_boxes = np.zeros_like(boxes)
        
        # Convert each box
        for i, box in enumerate(boxes):
            # Get polar coordinates
            r1, theta1, r2, theta2 = box
            
            # Denormalize radius if needed (assuming normalized by diagonal)
            if r1 <= 1.0 and r2 <= 1.0:
                r1 = r1 * diag
                r2 = r2 * diag
            
            # Calculate Cartesian coordinates
            x1 = center_x + r1 * np.cos(theta1)
            y1 = center_y + r1 * np.sin(theta1)
            x2 = center_x + r2 * np.cos(theta2)
            y2 = center_y + r2 * np.sin(theta2)
            
            # Ensure proper ordering (x1 <= x2, y1 <= y2)
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            
            # Store in output array
            cart_boxes[i] = [x_min, y_min, x_max, y_max]
        
        return cart_boxes
    
    def _polar_to_cartesian_landmarks(self, landmarks, image_size):
        """
        Convert landmarks from polar [r1, theta1, r2, theta2, ...] to Cartesian [x1, y1, x2, y2, ...] coordinates.
        
        Args:
            landmarks: Landmarks in polar coordinates [N, num_landmarks*2]
            image_size: Image size (height, width)
            
        Returns:
            cart_landmarks: Landmarks in Cartesian coordinates [N, num_landmarks*2]
        """
        height, width = image_size
        center_x, center_y = width / 2, height / 2
        diag = np.sqrt(width**2 + height**2)
        
        # Initialize output array
        cart_landmarks = np.zeros_like(landmarks)
        
        # Convert each set of landmarks
        for i, lm in enumerate(landmarks):
            # Convert each landmark
            for j in range(0, lm.shape[0], 2):
                # Get polar coordinates
                r, theta = lm[j], lm[j+1]
                
                # Skip invisible landmarks (r=0)
                if r == 0:
                    continue
                
                # Denormalize radius if needed (assuming normalized by diagonal)
                if r <= 1.0:
                    r = r * diag
                
                # Calculate Cartesian coordinates
                x = center_x + r * np.cos(theta)
                y = center_y + r * np.sin(theta)
                
                # Store in output array
                cart_landmarks[i, j] = x
                cart_landmarks[i, j+1] = y
        
        return cart_landmarks
