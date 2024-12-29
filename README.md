# CIFAR-10 Custom Network Training

![Build Status](./tests/status_badge.svg)

A PyTorch implementation of a custom CNN architecture for CIFAR-10 classification with:
- Strided convolutions
- Depthwise Separable Convolution
- Dilated Convolution
- Global Average Pooling
- Advanced augmentations using albumentations

## Requirements
- Python 3.10+
- PyTorch 1.9+
- CUDA 12.1 (for Linux/Windows GPU support)
- Apple Silicon / Intel support for MacOS
- Make utility
- Conda

## Quick Start

1. Setup environment and install dependencies:
```bash
make setup
```

2. Activate the environment:
```bash
make activate
# Run the command shown in the output
```

3. Train the model:
```bash
make train
```

4. Run tests:
```bash
# Run all tests
make test

# Run specific test suites
make test-architecture    # Test model architecture requirements
make test-augmentations  # Test data augmentation pipeline
```

5. View training logs:
```bash
make show-log
```

6. Clean up:
```bash
make clean
```

For a list of all available commands:
```bash
make help
```

## Project Structure
```
project_root/
├── models/
│   └── network.py         # Neural network architecture
├── utils/
│   ├── transforms.py      # Data transformations
│   └── logger.py          # Training logger
├── datasets/
│   └── custom_dataset.py  # Dataset class
├── checkpoints/          # Model checkpoints
│   ├── last_checkpoint.pth  # Latest training state
│   └── best_model.pth      # Best performing model
├── tests/
│   ├── test_model_architecture.py  # Architecture tests
│   ├── test_augmentations.py      # Augmentation tests
│   └── status_badge.svg           # Local build status
├── train.py               # Training script
├── requirements.txt       # Python dependencies
└── Makefile              # Build automation
```

## Model Requirements

The implementation must satisfy the following requirements:

- [ ] Architecture follows C1C2C3C40 pattern (No MaxPooling)
- [ ] Uses 3 3x3 layers with stride of 2 instead of MaxPooling
- [ ] Total receptive field > 44
- [ ] Contains Depthwise Separable Convolution
- [ ] Contains Dilated Convolution
- [ ] Uses Global Average Pooling
- [ ] Total parameters < 200k
- [ ] Achieves 85% accuracy

### Data Augmentation Requirements

The training pipeline includes:
- [ ] Horizontal flip
- [ ] ShiftScaleRotate
- [ ] CoarseDropout with specified parameters:
  - max_holes = 1
  - max_height = 16px
  - max_width = 1
  - min_holes = 1
  - min_height = 16px
  - min_width = 16px
  - fill_value = dataset mean