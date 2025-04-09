import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import multiprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
import random
import math  # Added for cosine scheduler

# --- Configuration ---
# Set the path to the dataset directory
# PLEASE UPDATE THIS PATH TO YOUR ACTUAL DATASET LOCATION
data_dir = "/home/rikisu/NNDL/CNN/cell_images"  # <--- MAKE SURE THIS PATH IS CORRECT

# Training configuration
epochs = 50  # Increased epochs to give more training time
batch_size = 32
learning_rate = 0.0003  # Lower initial learning rate
seed = 42  # For reproducibility
patience = 15  # Increased patience for early stopping

# Automatically set the number of worker processes for data loading
num_workers = multiprocessing.cpu_count()
print(f"Using {num_workers} workers for data loading.")

# Directory to save results and models
save_dir = 'results'
os.makedirs(save_dir, exist_ok=True)
best_model_save_path = os.path.join(save_dir, 'best_model_Team5.pth')

# --- Reproducibility ---
print(f"Setting random seed to: {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True

# Use a fixed generator for reproducible splits
generator = torch.Generator().manual_seed(seed)

# --- Data Transformations ---
# Enhanced augmentation for the training set
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),  # Added vertical flipping
    transforms.RandomRotation(30),  # Increased rotation range
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added affine transforms
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Increased jitter values
    transforms.RandomGrayscale(p=0.05),  # Occasionally convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# No augmentation for validation and test sets
val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Load and Split Dataset ---
try:
    # Load the full dataset *metadata* first (without applying transforms yet)
    full_dataset_no_transform = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset_no_transform.classes
    num_classes = len(class_names)
    print(f"Dataset loaded successfully. Found {len(full_dataset_no_transform)} images in {num_classes} classes: {class_names}")

    # Define split sizes
    total_size = len(full_dataset_no_transform)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    print(f"Splitting dataset into: Train={train_size}, Validation={val_size}, Test={test_size}")

    # Split indices using the generator for reproducibility
    indices = list(range(total_size))
    train_indices, val_indices, test_indices = random_split(
        indices, [train_size, val_size, test_size], generator=generator
    )

    # Create Subset instances and assign the correct transforms
    train_dataset = Subset(full_dataset_no_transform, train_indices.indices)
    train_dataset.dataset.transform = train_transform  # Assign transform to the underlying dataset for this subset

    val_dataset = Subset(full_dataset_no_transform, val_indices.indices)
    val_dataset.dataset.transform = val_test_transform  # Assign transform

    test_dataset = Subset(full_dataset_no_transform, test_indices.indices)
    test_dataset.dataset.transform = val_test_transform  # Assign transform

except FileNotFoundError:
    print(f"ERROR: Dataset directory not found at '{data_dir}'. Please ensure the path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred loading or splitting the dataset: {e}")
    exit()


# --- Create Data Loaders ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# --- Define Enhanced CNN Model with Deeper Architecture ---
class MalariaCNNImproved(nn.Module):
    def __init__(self, num_classes=2):
        super(MalariaCNNImproved, self).__init__()
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)  # Added second conv
        self.bn1_2 = nn.BatchNorm2d(32)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Added second conv
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)  # Added second conv
        self.bn3_2 = nn.BatchNorm2d(128)
        self.relu3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # New Block 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.relu4_1 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flattened size is now adjusted for the extra pooling layer
        self.flattened_size = 256 * 8 * 8  # After 4 pooling layers: 128/16 = 8
        
        # FC layers
        self.fc1 = nn.Linear(self.flattened_size, 512)  # Increased neurons
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        # Add a second FC layer
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)  # Slightly less dropout in final layer
        
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # Block 4
        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.relu5(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.relu6(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MalariaCNNImproved(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # Changed to AdamW with weight decay

# --- Learning Rate Scheduler with Warmup ---
def warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = warmup_cosine_schedule(optimizer, warmup_epochs=5, total_epochs=epochs)

# Mixed Precision Scaler
use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
print(f"Using Automatic Mixed Precision (AMP): {use_amp}")


print("\n--- Starting Training ---")
train_losses = []
val_losses = []
val_accuracies = []

# Variables for Early Stopping and Best Model Saving - now tracking accuracy instead of loss
best_val_accuracy = 0.0
patience_counter = 0

for epoch in range(epochs):    
    torch.cuda.empty_cache()
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Use autocast context manager for forward pass with AMP
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale loss, backward pass, and optimizer step
        scaler.scale(loss).backward()
        # Optional: Gradient Clipping
        scaler.unscale_(optimizer)  # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")  # Show LR

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- Validation Phase ---
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

    with torch.no_grad():
        for images, labels in val_progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Autocast for validation forward pass too
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_running_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs} => Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.2f}%")

    # --- Learning Rate Scheduling ---
    scheduler.step()  # Step the scheduler based on epoch (not loss)

    # --- Early Stopping & Best Model Check - now based on accuracy ---
    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        patience_counter = 0
        print(f"  Val accuracy improved to {best_val_accuracy:.2f}%. Saving model to {best_model_save_path}")
        try:
            torch.save(model.state_dict(), best_model_save_path)
        except Exception as e:
            print(f"  Error saving model: {e}")
    else:
        patience_counter += 1
        print(f"  Val accuracy did not improve. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"--- Early stopping triggered after epoch {epoch+1} ---")
            break  # Exit training loop

print("\n--- Training Finished ---")


# --- Final Test Set Evaluation ---
print(f"\n--- Loading best model from {best_model_save_path} for final evaluation ---")
try:
    model.load_state_dict(torch.load(best_model_save_path))
    model.to(device)  # Ensure model is on correct device after loading
except FileNotFoundError:
    print(f"ERROR: Best model file not found at {best_model_save_path}. Evaluating with the last model state.")
except Exception as e:
    print(f"Error loading best model state: {e}. Evaluating with the last model state.")

model.eval()  # Set model to evaluation mode

test_correct = 0
test_total = 0
test_all_preds = []
test_all_labels = []
test_all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="[Test Eval]"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Use autocast for consistency
        with torch.amp.autocast("cuda", enabled=use_amp):
             outputs = model(images)

        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        test_all_preds.extend(predicted.cpu().numpy())
        test_all_labels.extend(labels.cpu().numpy())
        if num_classes == 2:
             test_all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class

# --- Calculate Final Metrics ---
test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0

avg_method = 'binary'
test_precision = precision_score(test_all_labels, test_all_preds, average=avg_method, zero_division=0)
test_recall = recall_score(test_all_labels, test_all_preds, average=avg_method, zero_division=0)
test_f1 = f1_score(test_all_labels, test_all_preds, average=avg_method, zero_division=0)

test_auc = 0.0  # Initialize AUC
unique_labels = np.unique(test_all_labels)
if num_classes == 2 and len(unique_labels) > 1 and len(test_all_probs) == len(test_all_labels):
    try:
        test_auc = roc_auc_score(test_all_labels, test_all_probs)
    except ValueError as e:
        print(f"Could not calculate AUC: {e}. Check if probabilities align with labels. Assigning AUC = 0.0")
        test_auc = 0.0


print(f"\n--- Test Set Results (using best model) ---")
print(f"Accuracy:  {test_accuracy:.2f}%")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"AUC:       {test_auc:.4f}") 

cm = confusion_matrix(test_all_labels, test_all_preds)
print("\nConfusion Matrix:")
print(cm)


# --- Store Results for Plotting ---
results = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'test_auc': test_auc, 
    'confusion_matrix': cm
}

# --- Visualization (Adjusted y-limit for metrics bar) ---
def plot_results(results_dict, class_names_list, save_path):
    num_epochs_run = len(results_dict.get('train_losses', []))  # Get actual epochs run
    epochs_axis = range(1, num_epochs_run + 1)

    plt.figure(figsize=(16, 12))  # Slightly larger figure

    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_axis, results_dict.get('train_losses', []), label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs_axis, results_dict.get('val_losses', []), label='Validation Loss', marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs_axis)  # Ensure integer ticks for epochs

    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs_axis, results_dict.get('val_accuracies', []), label='Validation Accuracy', color='g', marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs_axis)

    # Plot confusion matrix
    cm_plot = results_dict.get('confusion_matrix')
    if cm_plot is not None and len(class_names_list) > 0:
        plt.subplot(2, 2, 3)
        im = plt.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Test Set)')
        plt.colorbar(im, fraction=0.046, pad=0.04)  # Adjust colorbar size
        tick_marks = np.arange(len(class_names_list))
        plt.xticks(tick_marks, class_names_list, rotation=45, ha="right")  # Improve label readability
        plt.yticks(tick_marks, class_names_list)

        # Add text annotations
        thresh = cm_plot.max() / 2.
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                plt.text(j, i, format(cm_plot[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm_plot[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Plot additional test metrics bar chart
    plt.subplot(2, 2, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    values = [
        results_dict.get('test_accuracy', 0) / 100.0,  # Scale accuracy
        results_dict.get('test_precision', 0),
        results_dict.get('test_recall', 0),
        results_dict.get('test_f1', 0),
        results_dict.get('test_auc', 0)
    ]
    bars = plt.bar(metrics, values, color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e'])
    plt.title('Final Test Set Metrics (Best Model)')
    plt.ylabel('Score')
    plt.ylim(0.9, 1.1) 

    # Add text labels above bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontsize=9)

    plt.suptitle('Malaria Cell Classification Results (Team 5 Improved)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout considering suptitle

    # Save the plot
    plot_filename = os.path.join(save_path, 'training_validation_test_results_Team5_improved.png')
    try:
        plt.savefig(plot_filename)
        print(f"\nResults plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close()  # Close the plot

# Call the plotting function
plot_results(results, class_names, save_dir)

print("\n--- Script Finished ---")