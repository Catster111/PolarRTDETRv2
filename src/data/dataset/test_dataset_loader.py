"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Test script for WIDER Face dataset and polar face functionality dataห
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")

try:
    import torch
    import torchvision
    import numpy as np
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    print("✅ Basic imports successful")
except ImportError as e:
    print(f"❌ Basic import error: {e}")
    sys.exit(1)

# Try importing project modules
try:
    from src.core import register, GLOBAL_CONFIG
    from src.data.dataset.wider_face_dataset import WiderFaceDetection
    from src.data.transforms import Compose
    from src.data.dataloader import FaceLandmarkCollateFuncion
    print("✅ Project imports successful")
except ImportError as e:
    print(f"❌ Project import error: {e}")
    print("Available modules in src.data.dataset:")
    try:
        import src.data.dataset
        print([f for f in dir(src.data.dataset) if not f.startswith('_')])
    except:
        print("Could not import src.data.dataset")
    sys.exit(1)


def test_wider_face_dataset():
    """Test WIDER Face dataset loading"""
    print("\n🧪 Testing WIDER Face Dataset...")
    
    # Check if dataset exists
    dataset_root = project_root / "dataset" / "wider_face"
    if not dataset_root.exists():
        print(f"❌ Dataset not found at {dataset_root}")
        print("Please download WIDER Face dataset first")
        return False
    
    try:
        # Create dataset
        train_root = dataset_root / "WIDER_train" / "images"
        train_ann = dataset_root / "wider_face_split" / "wider_face_train_bbx_gt.txt"
        
        if not train_root.exists() or not train_ann.exists():
            print(f"❌ Missing dataset files:")
            print(f"   Images: {train_root.exists()}")
            print(f"   Annotations: {train_ann.exists()}")
            return False
        
        # Test dataset creation
        dataset = WiderFaceDetection(
            root=str(train_root),
            ann_file=str(train_ann),
            image_set='train',
            return_landmarks=True,
            min_face_size=16,
            use_synthetic_landmarks=True
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   Total samples: {len(dataset)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            image, target = sample
            
            print(f"   Sample 0:")
            print(f"     Image size: {image.size}")
            print(f"     Number of faces: {len(target['boxes'])}")
            print(f"     Has landmarks: {'landmarks' in target}")
            print(f"     Has polar features: {'polar_features' in target}")
            
            return True
        else:
            print("❌ Dataset is empty")
            return False
            
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transforms():
    """Test data transforms"""
    print("\n🧪 Testing Data Transforms...")
    
    try:
        # Create dummy data
        image = Image.new('RGB', (640, 480), color='red')
        target = {
            'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            'labels': torch.tensor([0], dtype=torch.long),
            'landmarks': torch.tensor([[120, 120, 180, 120, 150, 140, 130, 160, 170, 160]], dtype=torch.float32),
            'image_id': torch.tensor([1]),
            'area': torch.tensor([10000.0]),
            'iscrowd': torch.tensor([0]),
            'orig_size': torch.tensor([640, 480])
        }
        
        # Test basic transforms
        from src.data.transforms import (
            RandomHorizontalFlip, Resize, ConvertPILImage, ConvertBoxes
        )
        
        transforms = Compose([
            Resize(size=[640, 640]),
            ConvertPILImage(dtype='float32', scale=True),
            ConvertBoxes(fmt='cxcywh', normalize=True)
        ])
        
        # Apply transforms
        transformed = transforms(image, target, None)
        
        print("✅ Transforms working correctly")
        print(f"   Input image size: {image.size}")
        print(f"   Output image shape: {transformed[0].shape}")
        print(f"   Output boxes shape: {transformed[1]['boxes'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collate_function():
    """Test face landmark collate function"""
    print("\n🧪 Testing Face Landmark Collate Function...")
    
    try:
        # Create dummy batch data
        batch_data = []
        for i in range(2):
            image = torch.randn(3, 640, 640)
            target = {
                'boxes': torch.tensor([[0.3, 0.3, 0.7, 0.7]], dtype=torch.float32),
                'labels': torch.tensor([0], dtype=torch.long),
                'landmarks': torch.tensor([[200, 200, 400, 200, 300, 250, 250, 300, 350, 300]], dtype=torch.float32),
                'image_id': torch.tensor([i]),
                'area': torch.tensor([0.16]),
                'iscrowd': torch.tensor([0]),
                'orig_size': torch.tensor([640, 640])
            }
            batch_data.append((image, target))
        
        # Test collate function
        collate_fn = FaceLandmarkCollateFuncion(
            polar_bins=36,
            heatmap_size=64,
            landmark_sigma=2.0
        )
        
        images, targets = collate_fn(batch_data)
        
        print("✅ Collate function working correctly")
        print(f"   Batch image shape: {images.shape}")
        print(f"   Number of targets: {len(targets)}")
        print(f"   Has landmark heatmaps: {'landmark_heatmaps' in targets[0]}")
        print(f"   Has polar features: {'polar_features' in targets[0]}")
        
        if 'landmark_heatmaps' in targets[0]:
            print(f"   Heatmap shape: {targets[0]['landmark_heatmaps'].shape}")
        
        if 'polar_features' in targets[0]:
            polar = targets[0]['polar_features']
            print(f"   Polar angle bin: {polar['polar_angle_bin']}")
            print(f"   Polar magnitude: {polar['polar_magnitude']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Collate function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_polar_face_model():
    """Test polar face model components"""
    print("\n🧪 Testing Polar Face Model Components...")
    
    try:
        from src.zoo.rtdetr.polar_face_decoder import PolarFaceTransformer
        from src.zoo.rtdetr.polar_face_criterion import PolarFaceCriterion
        from src.zoo.rtdetr.polar_face_postprocessor import PolarFacePostProcessor
        
        # Test model creation
        model = PolarFaceTransformer(
            num_classes=1,
            hidden_dim=256,
            num_queries=100,
            num_landmarks=5,
            polar_bins=36,
            heatmap_size=64
        )
        
        print("✅ Polar Face Transformer created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        dummy_feats = [
            torch.randn(2, 512, 80, 80),
            torch.randn(2, 1024, 40, 40),
            torch.randn(2, 2048, 20, 20)
        ]
        
        outputs = model(dummy_feats)
        
        print("✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        print(f"   Pred logits shape: {outputs['pred_logits'].shape}")
        print(f"   Pred boxes shape: {outputs['pred_boxes'].shape}")
        print(f"   Pred landmarks shape: {outputs['pred_landmarks'].shape}")
        print(f"   Landmark heatmaps shape: {outputs['landmark_heatmaps'].shape}")
        
        # Test criterion
        from src.zoo.rtdetr.matcher import HungarianMatcher
        
        matcher = HungarianMatcher(
            weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
            alpha=0.25,
            gamma=2.0
        )
        
        criterion = PolarFaceCriterion(
            matcher=matcher,
            weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_landmarks': 10},
            losses=['vfl', 'boxes', 'landmarks'],
            num_landmarks=5,
            polar_bins=36
        )
        
        # Create dummy targets
        targets = [
            {
                'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.4]], dtype=torch.float32),
                'labels': torch.tensor([0], dtype=torch.long),
                'landmarks': torch.tensor([[0.4, 0.4, 0.6, 0.4, 0.5, 0.5, 0.45, 0.6, 0.55, 0.6]], dtype=torch.float32),
            }
            for _ in range(2)
        ]
        
        losses = criterion(outputs, targets)
        
        print("✅ Criterion working correctly")
        print(f"   Loss keys: {list(losses.keys())}")
        
        # Test postprocessor
        postprocessor = PolarFacePostProcessor(
            num_classes=1,
            num_landmarks=5,
            polar_bins=36
        )
        
        orig_target_sizes = torch.tensor([[640, 640], [640, 640]], dtype=torch.float32)
        results = postprocessor(outputs, orig_target_sizes)
        
        print("✅ Postprocessor working correctly")
        print(f"   Number of results: {len(results)}")
        print(f"   Result keys: {list(results[0].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration Loading...")
    
    try:
        from src.core import YAMLConfig
        
        # Check if config exists
        config_path = project_root / "configs" / "polar_face" / "polar_face_wider_train.yml"
        
        if config_path.exists():
            cfg = YAMLConfig(str(config_path))
            print("✅ Configuration loaded successfully")
            print(f"   Task: {cfg.yaml_cfg.get('task')}")
            print(f"   Model: {cfg.yaml_cfg.get('model')}")
            print(f"   Criterion: {cfg.yaml_cfg.get('criterion')}")
            print(f"   Epochs: {cfg.yaml_cfg.get('epoches')}")
            return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_sample():
    """Visualize a sample with landmarks and polar features"""
    print("\n🧪 Testing Sample Visualization...")
    
    try:
        # Create dummy sample for visualization
        image = Image.new('RGB', (640, 480), color=(128, 128, 128))
        draw = ImageDraw.Draw(image)
        
        # Draw a face
        face_box = [200, 150, 400, 350]
        draw.rectangle(face_box, outline='red', width=3)
        
        # Draw landmarks (5 points)
        landmarks = [
            (250, 200),  # left eye
            (350, 200),  # right eye
            (300, 230),  # nose
            (270, 280),  # left mouth
            (330, 280),  # right mouth
        ]
        
        for i, (x, y) in enumerate(landmarks):
            draw.ellipse([x-5, y-5, x+5, y+5], fill='blue')
            draw.text((x+10, y), f'L{i+1}', fill='blue')
        
        # Calculate polar features
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        eye_vector = right_eye - left_eye
        angle = np.arctan2(eye_vector[1], eye_vector[0])
        angle_degrees = np.degrees(angle)
        
        # Draw orientation vector
        center = (left_eye + right_eye) / 2
        end_point = center + eye_vector * 0.5
        draw.line([tuple(center), tuple(end_point)], fill='green', width=3)
        
        # Add text info
        draw.text((50, 50), f'Face Orientation: {angle_degrees:.1f}°', fill='white')
        draw.text((50, 70), f'Eye Distance: {np.linalg.norm(eye_vector):.1f}px', fill='white')
        
        # Save visualization
        output_path = project_root / "test_visualization.jpg"
        image.save(output_path)
        
        print("✅ Sample visualization created")
        print(f"   Saved to: {output_path}")
        print(f"   Face angle: {angle_degrees:.1f} degrees")
        print(f"   Eye distance: {np.linalg.norm(eye_vector):.1f} pixels")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Polar Face Dataset and Components')
    parser.add_argument('--test', choices=['all', 'dataset', 'transforms', 'collate', 'model', 'config', 'viz'], 
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    print("🚀 Starting Polar Face Component Tests")
    print("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    if args.test in ['all', 'dataset']:
        total_tests += 1
        if test_wider_face_dataset():
            success_count += 1
    
    if args.test in ['all', 'transforms']:
        total_tests += 1
        if test_transforms():
            success_count += 1
    
    if args.test in ['all', 'collate']:
        total_tests += 1
        if test_collate_function():
            success_count += 1
    
    if args.test in ['all', 'model']:
        total_tests += 1
        if test_polar_face_model():
            success_count += 1
    
    if args.test in ['all', 'config']:
        total_tests += 1
        if test_config_loading():
            success_count += 1
    
    if args.test in ['all', 'viz']:
        total_tests += 1
        if visualize_sample():
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Ready for polar face training.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()