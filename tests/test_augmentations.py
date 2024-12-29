import pytest
from utils.transforms import get_transforms
import albumentations as A
import numpy as np
import torch

def test_augmentation_pipeline():
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
    train_transform, _ = get_transforms(MEAN, STD)
    
    # Check for horizontal flip
    has_hflip = any(isinstance(t, A.HorizontalFlip) for t in train_transform)
    assert has_hflip, "Transforms must include HorizontalFlip"
    
    # Check for ShiftScaleRotate
    has_ssr = any(isinstance(t, A.ShiftScaleRotate) for t in train_transform)
    assert has_ssr, "Transforms must include ShiftScaleRotate"

def test_transform_output():
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
    train_transform, _ = get_transforms(MEAN, STD)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    # Apply transformation
    transformed = train_transform(image=dummy_image)
    
    # Check if output is as expected
    assert 'image' in transformed, "Transform should return dictionary with 'image' key"
    assert isinstance(transformed['image'], torch.Tensor), "Transformed image should be a torch.Tensor"
    assert transformed['image'].shape == (3, 32, 32), "Image should be CHW format with size 32x32" 