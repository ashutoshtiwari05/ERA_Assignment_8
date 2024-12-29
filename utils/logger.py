import csv
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Logger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.csv')
        
        # Initialize CSV file with detailed headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Epoch', 
                'Train Loss', 
                'Train Accuracy',
                'Test Loss',
                'Test Accuracy',
                'Learning Rate',
                'Best Test Accuracy',
                'Time Taken (s)'
            ])
        self.best_test_acc = 0
    
    def log_epoch(self, epoch, train_metrics, test_metrics, learning_rate, epoch_time):
        self.best_test_acc = max(self.best_test_acc, test_metrics['accuracy'])
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{train_metrics["loss"]:.4f}',
                f'{train_metrics["accuracy"]:.2f}',
                f'{test_metrics["loss"]:.4f}',
                f'{test_metrics["accuracy"]:.2f}',
                f'{learning_rate:.6f}',
                f'{self.best_test_acc:.2f}',
                f'{epoch_time:.2f}'
            ])
    
    def get_log_file(self):
        return self.log_file

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(data_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }