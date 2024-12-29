import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from models.network import CustomNet
from utils.transforms import get_transforms
from utils.logger import Logger, evaluate_model
from datasets.custom_dataset import CustomDataset
import numpy as np
from tqdm import tqdm
import time
import os
import json

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints'):
    """Save checkpoint and best model"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # If model is DataParallel, save the module's state dict instead
    if isinstance(state['model_state_dict'], nn.DataParallel):
        state['model_state_dict'] = state['model_state_dict'].module.state_dict()
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_model_path)
        print(f"Best model saved: {best_model_path}")

def load_checkpoint(model, optimizer, scheduler, device, checkpoint_dir='checkpoints'):
    """Load checkpoint if exists"""
    checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
    start_epoch = 0
    best_acc = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=device
        )
        
        state_dict = checkpoint['model_state_dict']
        
        # If current model is DataParallel but checkpoint isn't
        if isinstance(model, nn.DataParallel) and not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        
        # If current model isn't DataParallel but checkpoint is
        elif not isinstance(model, nn.DataParallel) and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k in state_dict.items()}
            
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        
        # Move optimizer states to GPU if using CUDA
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
                        
        print(f"Resuming from epoch {start_epoch} with best accuracy: {best_acc:.2f}%")
    
    return start_epoch, best_acc

def train_model(train_loader, model, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{accuracy:.2f}%'
        })
    
    return {
        'loss': running_loss / len(train_loader),
        'accuracy': accuracy
    }

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        if device.type == 'cuda':  # Only use autocast for CUDA
            with torch.amp.autocast(device_type=device.type):
                for images, labels in test_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
        else:  # CPU evaluation
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return {
        'loss': running_loss / len(test_loader),
        'accuracy': accuracy
    }

def main():
    # Set up device and optimization flags
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        scaler = torch.amp.GradScaler()
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU")
        scaler = None
    
    # Initialize logger
    logger = Logger()
    
    # Training parameters
    num_epochs = 300  # Increased from 200
    batch_size = 128  # Adjusted for better gradient estimates
    
    # CIFAR-10 statistics
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
    
    # Get transforms
    train_transform, test_transform = get_transforms(MEAN, STD)
    
    # Download CIFAR-10 dataset
    cifar_train = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True
    )
    
    cifar_test = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True
    )
    
    # Create datasets
    train_dataset = CustomDataset(
        data=cifar_train.data,
        targets=np.array(cifar_train.targets),
        transform=train_transform
    )
    
    test_dataset = CustomDataset(
        data=cifar_test.data,
        targets=np.array(cifar_test.targets),
        transform=test_transform
    )
    
    # Optimize DataLoader settings
    num_workers = min(8, os.cpu_count())  # Optimize worker count
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # Can use larger batch size for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create model and move to device
    model = CustomNet(num_classes=10, device=device)
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Ensure model is on GPU
    if next(model.parameters()).device.type != 'cuda' and torch.cuda.is_available():
        print("Warning: Model is not on GPU! Moving to GPU...")
        model = model.cuda()
    
    # Move criterion to GPU
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    
    # Loss and optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Use CosineAnnealingLR with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # First restart at epoch 20
        T_mult=2,  # Each new restart is twice as long
        eta_min=1e-6
    )
    
    # Load checkpoint if exists - pass device parameter
    start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    print(f'Training logs will be saved to: {logger.get_log_file()}')
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch: {epoch+1}/{num_epochs}')
        epoch_start = time.time()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_metrics = train_model(
            train_loader, model, criterion, optimizer, device, scaler
        )
        
        # Test
        test_metrics = evaluate_model(
            model, test_loader, criterion, device
        )
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Check if this is the best model
        is_best = test_metrics['accuracy'] > best_acc
        if is_best:
            best_acc = test_metrics['accuracy']
        
        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
        }, is_best)
        
        # Log metrics
        logger.log_epoch(
            epoch + 1,
            train_metrics,
            test_metrics,
            current_lr,
            epoch_time
        )
        
        # Print metrics
        print(f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.2f}%')
        print(f'Test Loss: {test_metrics["loss"]:.4f}, Test Acc: {test_metrics["accuracy"]:.2f}%')
        print(f'Time: {epoch_time:.2f}s, LR: {current_lr:.6f}')
        print(f'Best Acc: {best_acc:.2f}%')
        
        # Update learning rate
        scheduler.step()

if __name__ == '__main__':
    main() 