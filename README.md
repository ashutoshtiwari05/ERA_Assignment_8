# CIFAR-10 Custom Network Training

![Build Status](./tests/status_badge.svg)

A PyTorch implementation of a custom CNN architecture for CIFAR-10 classification that demonstrates modern deep learning techniques and best practices.

## Architecture Features

- **Efficient Downsampling**: Uses strided convolutions instead of MaxPooling
- **Advanced Convolution Types**:
  - Depthwise Separable Convolution for parameter efficiency
  - Dilated Convolution for expanded receptive field
  - Global Average Pooling for spatial dimension reduction
- **Modern Training Components**:
  - Label smoothing
  - Cosine annealing with warm restarts
  - Mixed precision training
  - Gradient clipping
  - Advanced data augmentations

## Technical Requirements

### Model Architecture
- [x] C1C2C3C40 pattern without MaxPooling
- [x] 3 strided convolutions (stride=2) for downsampling
- [x] Receptive field > 44
- [x] Includes Depthwise Separable Convolution
- [x] Includes Dilated Convolution
- [x] Uses Global Average Pooling
- [x] Parameters < 200k
- [ ] Target accuracy: 85%

### Data Augmentation Pipeline
Advanced augmentations using albumentations:
- [x] Horizontal flip
- [x] ShiftScaleRotate
- [x] CoarseDropout
- [x] Color augmentations
- [x] Noise augmentations

## System Requirements
- Python 3.10+
- PyTorch 1.9+
- CUDA 12.1 (optional, for GPU support)
- Apple Silicon / Intel support for MacOS
- Make utility
- Conda package manager

## Quick Start

1. **Setup Environment**:
```bash
make setup
```

2. **Activate Environment**:
```bash
make activate
# Run the command shown in the output
```

3. **Train Model**:
```bash
make train
```

4. **Run Tests**:
```bash
make test                  # All tests
make test-architecture    # Model architecture tests
make test-augmentations  # Augmentation tests
```

5. **Monitor Training**:
```bash
make show-log
```

## Project Structure

```
project_root/
├── .github/
│   └── workflows/         # CI/CD configurations
├── models/
│   └── network.py        # Neural network architecture
├── utils/
│   ├── transforms.py     # Data augmentation pipeline
│   └── logger.py        # Training metrics logger
├── datasets/
│   └── custom_dataset.py # Dataset implementation
├── tests/
│   ├── test_model_architecture.py
│   ├── test_augmentations.py
│   └── status_badge.svg  # Test coverage badge
├── checkpoints/          # Model weights
├── logs/                 # Training logs
├── train.py             # Training script
├── requirements.txt      # Dependencies
└── Makefile             # Build automation
```

## Implementation Details

### Network Architecture
- Input → C1 block → C2 block → C3 block → C4 block → GAP → Output
- Each block contains specific convolution types and operations
- Total parameters: ~150k
- Receptive field: >60 pixels

### Training Pipeline
- Mixed precision training (when CUDA available)
- Gradient clipping at 1.0
- Label smoothing (0.1)
- AdamW optimizer with weight decay
- CosineAnnealingWarmRestarts scheduler
- Automatic checkpointing and model saving

### Data Pipeline
- Efficient data loading with:
  - Pinned memory
  - Persistent workers
  - Prefetching
  - Optimal worker count
- Comprehensive augmentation strategy
- Normalized inputs (CIFAR-10 mean/std)

## Development

### Running Tests
Tests validate both model architecture requirements and training components:

```bash
# Run all tests with coverage
make test

# Run specific test suites
make test-architecture
make test-augmentations
```

### CI/CD Pipeline
GitHub Actions automatically:
- Runs all tests on push/PR
- Generates test coverage badge
- Validates model architecture requirements
- Ensures code quality

## License
MIT License - see LICENSE file for details