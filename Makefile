.PHONY: setup clean train help show-log activate test test-architecture test-augmentations

# Variables
CONDA_ENV_NAME = era-assignment-8
PYTHON_VERSION = 3.10
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
PYTHONPATH := $(shell pwd)
export PYTHONPATH

# Detect OS and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# PyTorch install command based on platform
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        # MacOS ARM (M1/M2)
        PYTORCH_INSTALL = conda install pytorch torchvision -c pytorch
    else
        # MacOS Intel
        PYTORCH_INSTALL = conda install pytorch torchvision -c pytorch
    endif
else ifeq ($(UNAME_S),Linux)
    # Check if NVIDIA GPU is available
    NVIDIA_GPU := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo "yes" || echo "no")
    ifeq ($(NVIDIA_GPU),yes)
        # Linux with CUDA
        PYTORCH_INSTALL = conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    else
        # Linux without CUDA
        PYTORCH_INSTALL = conda install pytorch torchvision cpuonly -c pytorch
    endif
else
    # Windows or other OS - default to CPU version
    PYTORCH_INSTALL = conda install pytorch torchvision cpuonly -c pytorch
endif

help:
	@echo "Available commands:"
	@echo "make setup      - Create conda environment and install all dependencies"
	@echo "make activate   - Activate the conda environment"
	@echo "make train      - Run the training script"
	@echo "make clean      - Remove Python cache files and data"
	@echo "make show-log   - Show the latest training log"

setup:
	@echo "Creating conda environment..."
	mkdir -p checkpoints logs
	conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION) -y
	@echo "Installing PyTorch..."
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && \
	$(PYTORCH_INSTALL) && \
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && \
	pip install -r requirements.txt
	@echo "Setup complete. Use 'make activate' to activate the environment"

activate:
	@echo "To activate the environment, run:"
	@echo "source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate $(CONDA_ENV_NAME)"

train:
	@if [ -z "$${CONDA_DEFAULT_ENV}" ] || [ "$${CONDA_DEFAULT_ENV}" != "$(CONDA_ENV_NAME)" ]; then \
		echo "Error: Please activate the environment first using 'make activate'"; \
		exit 1; \
	fi
	@echo "Starting training..."
	@python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
	python train.py

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf data/
	@echo "Note: Checkpoints and logs are preserved. Use 'make clean-all' to remove them."

clean-all: clean
	rm -rf checkpoints/
	rm -rf logs/
	@echo "Cleanup complete"

show-log:
	@if [ -d "logs" ]; then \
		latest_log=$$(ls -t logs/*.csv 2>/dev/null | head -n1); \
		if [ -n "$$latest_log" ]; then \
			echo "Showing latest log: $$latest_log"; \
			cat "$$latest_log"; \
		else \
			echo "No log files found in logs directory"; \
		fi \
	else \
		echo "Logs directory not found"; \
	fi

test:
	PYTHONPATH=$(PYTHONPATH) pytest tests/ -v

test-architecture:
	PYTHONPATH=$(PYTHONPATH) pytest tests/test_model_architecture.py -v

test-augmentations:
	PYTHONPATH=$(PYTHONPATH) pytest tests/test_augmentations.py -v