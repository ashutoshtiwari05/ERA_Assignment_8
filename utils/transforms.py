import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(mean, std):
    train_transform = A.Compose([
        A.RandomResizedCrop(32, 32, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.5
        ),
        
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=30,
                sat_shift_limit=40,
                val_shift_limit=30,
                p=1.0
            ),
        ], p=0.5),
        
        A.OneOf([
            A.CoarseDropout(
                max_holes=3,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=[int(x * 255.0) for x in mean],
                p=1.0
            ),
            A.GridDropout(
                ratio=0.3,
                holes_number_x=4,
                holes_number_y=4,
                p=1.0
            ),
        ], p=0.5),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(p=1.0),
            A.MultiplicativeNoise(multiplier=[0.85, 1.15], per_channel=True, p=1.0),
        ], p=0.3),
        
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform, test_transform 