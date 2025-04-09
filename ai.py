import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import os
import time
import gc

# Configuration
CONFIG = {
    'batch_size': 16,
    'epochs': 40,
    'learning_rate': 0.001,
    'image_size': 128, 
    'data_dir': "/home/rikisu/NNDL/CNN/cell_images",
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'num_classes': 2,
    'dropout_rate': 0.3,
    'early_stopping_patience': 5,
    'lr_patience': 3,
    'lr_factor': 0.5,
    'save_dir': './models',
    'use_cross_validation': False,
    'n_folds': 5
}

# Create save directory if it doesn't exist
os.makedirs(CONFIG['save_dir'], exist_ok=True)

# Utilize available CPU cores
num_workers = multiprocessing.cpu_count()

# Define improved CNN model with residual blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ImprovedMalariaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ImprovedMalariaCNN, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)  # Reduced filters
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual blocks with fewer filters
        self.res_block1 = ResidualBlock(16, 32, stride=1)  # Reduced filters
        self.res_block2 = ResidualBlock(32, 64, stride=1)  # Reduced filters
        
        # Calculate the feature size after 2 max pooling operations (instead of 3)
        # Starting with 128x128, after 2 pooling operations: 128/(2^2) = 32
        feature_size = 64 * 32 * 32  # Reduced feature size
        
        # Simplified fully connected layers
        self.fc1 = nn.Linear(feature_size, 128)  # Reduced size
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(CONFIG['dropout_rate'])
        
        self.fc2 = nn.Linear(128, num_classes)  # Direct to output
        
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 128x128 -> 64x64
        
        # Residual blocks (only 2 instead of 3)
        x = self.res_block1(x)
        x = self.maxpool(x)  # 64x64 -> 32x32
        
        x = self.res_block2(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Simplified fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        
        return x

# Data augmentation for training
transform_train = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Just resize and normalize for validation/testing
transform_test = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
print(f"Loading dataset from {CONFIG['data_dir']}...")
dataset = datasets.ImageFolder(root=CONFIG['data_dir'], transform=transform_train)

# Check for class imbalance
class_names = dataset.classes
class_counts = [0, 0]
for _, label in dataset:
    class_counts[label] += 1
print(f"Class distribution: {class_names[0]}: {class_counts[0]}, {class_names[1]}: {class_counts[1]}")

# Calculate class weights for balanced loss
weights = torch.FloatTensor([1.0/class_counts[0], 1.0/class_counts[1]])
weights = weights / weights.sum() * 2  # Normalize and scale

# Define the training function
def train_and_evaluate(model, train_loader, val_loader, test_loader=None, fold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Using device: {device}")
    
    # Enable memory efficient features if using CUDA
    if device.type == 'cuda':
        # Empty cache before starting training
        torch.cuda.empty_cache()
        
        # Set memory allocation to reduce fragmentation
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    model = model.to(device)
    
    # Weighted loss function to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=CONFIG['lr_patience'], 
        factor=CONFIG['lr_factor']
    )
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_path = os.path.join(CONFIG['save_dir'], f"best_model{'_fold_'+str(fold) if fold is not None else ''}.pth")
    patience_counter = 0
    
    # Start training
    start_time = time.time()
    
    for epoch in range(CONFIG['epochs']):
        # Add garbage collection at the beginning of each epoch
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=False)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                
                # Get predictions
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for metrics calculation
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        # Additional metrics
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} completed:")
        print(f"  Train Loss: {epoch_train_loss:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Save best model and check for early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{CONFIG['early_stopping_patience']}")
            if patience_counter >= CONFIG['early_stopping_patience']:
                print("Early stopping triggered!")
                break
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Test on test set if provided
    if test_loader:
        print("\nEvaluating on test set...")
        test_correct = 0
        test_total = 0
        test_all_preds = []
        test_all_labels = []
        test_all_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    outputs = model(images)
                
                # Get predictions
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Store for metrics calculation
                test_all_preds.extend(predicted.cpu().numpy())
                test_all_labels.extend(labels.cpu().numpy())
                test_all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate test metrics
        test_accuracy = 100 * test_correct / test_total
        test_precision = precision_score(test_all_labels, test_all_preds, average='binary', zero_division=0)
        test_recall = recall_score(test_all_labels, test_all_preds, average='binary', zero_division=0)
        test_f1 = f1_score(test_all_labels, test_all_preds, average='binary', zero_division=0)
        test_auc = roc_auc_score(test_all_labels, test_all_probs)
        
        # Print test metrics
        print(f"Test Results:")
        print(f"  Accuracy: {test_accuracy:.2f}%")
        print(f"  Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}, AUC: {test_auc:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(test_all_labels, test_all_preds)
        print(f"Confusion Matrix:")
        print(cm)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'confusion_matrix': cm
        }
    else:
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss
        }

# Function to plot results
def plot_results(results):
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(results['train_losses'], label='Training Loss')
    plt.plot(results['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(results['val_accuracies'], label='Validation Accuracy', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    
    # Plot confusion matrix if available
    if 'confusion_matrix' in results:
        plt.subplot(2, 2, 3)
        cm = results['confusion_matrix']
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    # Plot additional metrics if available
    if 'test_accuracy' in results:
        plt.subplot(2, 2, 4)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        values = [
            results['test_accuracy']/100,  # Convert from percentage
            results['test_precision'],
            results['test_recall'],
            results['test_f1'],
            results['test_auc']
        ]
        plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
        plt.ylabel('Score')
        plt.ylim(0.9, 1.1) # Set y-limit slightly above 1.0
        plt.title('Test Metrics')
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['save_dir'], 'training_results_ai.png'))
    plt.close()

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if CONFIG['use_cross_validation']:
        print(f"Starting {CONFIG['n_folds']}-fold cross-validation...")
        kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
        fold_results = []
        
        # Prepare for cross-validation
        dataset.transform = transform_test  # Use non-augmented transforms for consistent splitting
        indices = list(range(len(dataset)))
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
            print(f"\n{'='*50}\nFold {fold+1}/{CONFIG['n_folds']}\n{'='*50}")
            
            # Split train into train and validation
            train_indices, val_indices = train_idx[:int(0.8*len(train_idx))], train_idx[int(0.8*len(train_idx)):]
            
            # Create data samplers
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            test_sampler = SubsetRandomSampler(test_idx)
            
            # Create data loaders
            train_dataset = datasets.ImageFolder(root=CONFIG['data_dir'], transform=transform_train)
            val_dataset = datasets.ImageFolder(root=CONFIG['data_dir'], transform=transform_test)
            test_dataset = datasets.ImageFolder(root=CONFIG['data_dir'], transform=transform_test)
            
            train_loader = DataLoader(
                train_dataset, batch_size=CONFIG['batch_size'], 
                sampler=train_sampler, num_workers=num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=CONFIG['batch_size'], 
                sampler=val_sampler, num_workers=num_workers, pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=CONFIG['batch_size'], 
                sampler=test_sampler, num_workers=num_workers, pin_memory=True
            )
            
            # Initialize model for this fold
            model = ImprovedMalariaCNN(num_classes=CONFIG['num_classes'])
            
            # Train and evaluate
            fold_result = train_and_evaluate(model, train_loader, val_loader, test_loader, fold=fold)
            fold_results.append(fold_result)
            
            # Plot results for this fold
            plot_results(fold_result)
        
        # Summarize cross-validation results
        print("\n{'='*50}\nCross-Validation Summary\n{'='*50}")
        avg_test_acc = np.mean([res['test_accuracy'] for res in fold_results])
        avg_test_f1 = np.mean([res['test_f1'] for res in fold_results])
        avg_test_auc = np.mean([res['test_auc'] for res in fold_results])
        
        print(f"Average Test Accuracy: {avg_test_acc:.2f}%")
        print(f"Average Test F1 Score: {avg_test_f1:.4f}")
        print(f"Average Test AUC: {avg_test_auc:.4f}")
        
    else:
        # Regular train/val/test split
        print("Performing standard train/validation/test split...")
        
        # Create splits
        dataset.transform = transform_test  # Use non-augmented transforms for consistent splitting
        train_size = int(CONFIG['train_split'] * len(dataset))
        val_size = int(CONFIG['val_split'] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Apply appropriate transforms
        train_dataset.dataset.transform = transform_train
        val_dataset.dataset.transform = transform_test
        test_dataset.dataset.transform = transform_test
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG['batch_size'], 
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=CONFIG['batch_size'], 
            shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=CONFIG['batch_size'], 
            shuffle=False, num_workers=num_workers, pin_memory=True
        )
        
        # Initialize model
        model = ImprovedMalariaCNN(num_classes=CONFIG['num_classes'])
        
        # Train and evaluate
        results = train_and_evaluate(model, train_loader, val_loader, test_loader)
        
        # Plot results
        plot_results(results)