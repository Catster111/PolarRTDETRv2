"""
Data prefetcher for Polar-RTDETRv2.

This module provides a data prefetcher implementation that moves data to GPU
asynchronously for faster training by overlapping data loading with computation.
"""

import torch
import torch.cuda.amp as amp


class data_prefetcher:
    """
    Data prefetcher that preloads data to GPU asynchronously.
    
    This class helps speed up training by overlapping data loading with computation.
    It preloads the next batch while the current batch is being processed.
    """
    
    def __init__(self, loader, device, use_amp=False):
        """
        Initialize the data prefetcher.
        
        Args:
            loader: Data loader
            device: Device to load data to (CPU or CUDA)
            use_amp: Whether to use automatic mixed precision
        """
        self.loader = iter(loader)
        self.device = device
        self.use_amp = use_amp
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self.next_images = None
        self.next_targets = None
        self.preload()
    
    def preload(self):
        """
        Preload the next batch of data to GPU.
        """
        try:
            self.next_images, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_images = None
            self.next_targets = None
            return
        
        # -------------------------------------------------------------
        # `DataLoader` might return a list/tuple of image tensors
        # (especially when the `collate_fn` simply zips the batch).
        # Convert such collection into a single stacked tensor so that
        # `.to(device)` can be called safely.
        # -------------------------------------------------------------
        if isinstance(self.next_images, (list, tuple)):
            # Ensure every element is a tensor
            imgs = [
                img if isinstance(img, torch.Tensor) else torch.as_tensor(img)
                for img in self.next_images
            ]

            # ---------------------------------------------------------
            # Object-detection datasets usually return variable-sized
            # images.  We pad each image to the maximum H × W in the
            # batch and create a contiguous tensor so that subsequent
            # `.to(device)` calls work without error.
            # ---------------------------------------------------------
            max_h = max(i.shape[-2] for i in imgs)
            max_w = max(i.shape[-1] for i in imgs)
            batch = torch.zeros(
                (len(imgs), imgs[0].shape[0], max_h, max_w),
                dtype=imgs[0].dtype,
            )
            for b, img in enumerate(imgs):
                _, h, w = img.shape
                batch[b, :, :h, :w] = img
            self.next_images = batch
        # If it's already a tensor we leave it untouched
        
        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                self.next_images = self.next_images.to(self.device, non_blocking=True)

                # ---------------------------------------------------------
                # Move *all* tensors inside the (potentially nested) target
                # structure to the same device to avoid mismatch errors
                # further in the pipeline (e.g. matcher / criterion).
                # ---------------------------------------------------------

                def _move_to_device(obj, device, non_blocking=False):
                    """
                    Recursively move tensors contained inside lists / tuples /
                    dicts to the specified device.
                    """
                    if torch.is_tensor(obj):
                        return obj.to(device, non_blocking=non_blocking)
                    if isinstance(obj, dict):
                        return {k: _move_to_device(v, device, non_blocking) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        converted = [_move_to_device(o, device, non_blocking) for o in obj]
                        return type(obj)(converted)  # preserve original type
                    return obj  # leave other types unchanged

                self.next_targets = _move_to_device(self.next_targets, self.device, non_blocking=True)
                
                # Apply AMP if needed
                if self.use_amp:
                    self.next_images = self.next_images.half()
    
    def next(self):
        """
        Get the next batch of data.
        
        Returns:
            images: Batch of images
            targets: Batch of targets
        """
        if self.device.type == 'cuda':
            torch.cuda.current_stream().wait_stream(self.stream)
        
        images = self.next_images
        targets = self.next_targets
        
        # Start preloading the next batch
        self.preload()
        
        return images, targets
