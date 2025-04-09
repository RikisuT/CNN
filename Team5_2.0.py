import torch
import torch.nn as nn
import torch.nn.functional as F # Added for softmax
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np # Added for metrics/plotting
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score # Added for detailed metrics
import os # Added for saving plot
import time

# Configuration
CONFIG = {
    'batch_size': 32, 
    'epochs': 40,      
    'learning_rate': 0.001, 
    'image_size': 128,      # Set image size
    'data_dir': "/home/rikisu/NNDL/CNN/cell_images", # Make sure this path is correct
    'train_split': 0.7,
    'val_split': 0.15,
    # 'test_split': 0.15 # Implicitly defined
    'num_classes': 2,
    'save_dir': './results' # Directory to save plots
}

# Create save directory if it doesn't exist
os.makedirs(CONFIG['save_dir'], exist_ok=True)

# Automatically set the number of worker processes for data loading
num_workers = multiprocessing.cpu_count()

# Define CNN model 
class MalariaCNN(nn.Module):
    def __init__(self, num_classes=2): # Added num_classes argument
        super(MalariaCNN, self).__init__()
        #---------- CONVOLUTIONAL LAYERS ----------#
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        #---------- FULLY CONNECTED LAYERS ----------#
        fc1_input_features = 128 * 16 * 16 # Calculated for 128x128 input, 3 pools
        self.fc1 = nn.Linear(fc1_input_features, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes) # Use num_classes

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x))) # 128 -> 64
        x = self.pool(self.relu2(self.conv2(x))) # 64 -> 32
        x = self.pool(self.relu3(self.conv3(x))) # 32 -> 16
        x = x.view(x.size(0), -1) # Flatten: [batch_size, 128*16*16]
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Data Transformations
transform_train = transforms.Compose([ # Added separate transform for potential augmentation later
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_val_test = transforms.Compose([ # Validation/Test transforms usually don't include augmentation
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset (only need to load once)
print(f"Loading dataset from {CONFIG['data_dir']}...")
try:
    # Load with a base transform first for splitting
    base_dataset = datasets.ImageFolder(root=CONFIG['data_dir'], transform=transform_val_test)
    class_names = base_dataset.classes # Get class names for plotting CM
    CONFIG['num_classes'] = len(class_names) # Update num_classes based on dataset
    print(f"Dataset loaded successfully. Found {len(base_dataset)} images in {CONFIG['num_classes']} classes: {class_names}")
except FileNotFoundError:
    print(f"Error: Dataset directory not found at {CONFIG['data_dir']}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()

# Create splits
total_size = len(base_dataset)
train_size = int(CONFIG['train_split'] * total_size)
val_size = int(CONFIG['val_split'] * total_size)
test_size = total_size - train_size - val_size

print(f"Splitting dataset: Train={train_size}, Validation={val_size}, Test={test_size}")
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    base_dataset, [train_size, val_size, test_size]
)

# IMPORTANT: Assign the correct transforms AFTER splitting
# Create wrapper datasets or handle transform assignment carefully.
# A common way is to access the underlying dataset object if using Subset.
# Since random_split returns Subset objects, we access their internal dataset.
# Note: This assigns the SAME transform object. If transforms had internal state,
# it might be better to create separate transform objects. For these standard
# transforms, it's usually fine.
train_dataset.dataset.transform = transform_train
val_dataset.dataset.transform = transform_val_test
test_dataset.dataset.transform = transform_val_test

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)
print(f"Using {num_workers} workers for data loading.")

#---------- MODEL SETUP ----------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MalariaCNN(num_classes=CONFIG['num_classes']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

# Mixed precision scaler
use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
print(f"Mixed precision training enabled: {scaler.is_enabled()}")

#---------- TRAINING & EVALUATION ----------#
train_losses = []
val_losses = []
val_accuracies = []

start_time = time.time()

print("Starting Training...")
for epoch in range(CONFIG['epochs']):
    torch.cuda.empty_cache()
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1e-6))

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # --- Validation Phase ---
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Validate]", leave=False)
        for images, labels in val_progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # Run validation in default precision or autocast if desired/needed
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_progress_bar.set_postfix(loss=val_running_loss / (val_progress_bar.n + 1e-6))


    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} Summary:")
    print(f"  Train Loss: {epoch_train_loss:.4f}")
    print(f"  Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2f}%")


training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# --- Final Test Set Evaluation ---
print("\nEvaluating on Test Set...")
model.eval()
test_correct = 0
test_total = 0
test_all_preds = []
test_all_labels = []
test_all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="[Test]"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)

        probs = F.softmax(outputs, dim=1) # Get probabilities for AUC
        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        test_all_preds.extend(predicted.cpu().numpy())
        test_all_labels.extend(labels.cpu().numpy())
        # Assuming binary classification, get probability of the positive class (class 1)
        if CONFIG['num_classes'] == 2:
             test_all_probs.extend(probs[:, 1].cpu().numpy())
        # Add handling here if num_classes > 2 and multi-class AUC is needed


# Calculate final test metrics
test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
# Use average='binary' for binary classification. Adjust if multi-class.
avg_method = 'binary' if CONFIG['num_classes'] == 2 else 'macro'
test_precision = precision_score(test_all_labels, test_all_preds, average=avg_method, zero_division=0)
test_recall = recall_score(test_all_labels, test_all_preds, average=avg_method, zero_division=0)
test_f1 = f1_score(test_all_labels, test_all_preds, average=avg_method, zero_division=0)
test_auc = None
if CONFIG['num_classes'] == 2 and len(np.unique(test_all_labels)) > 1: # AUC needs at least two classes present
    try:
        test_auc = roc_auc_score(test_all_labels, test_all_probs)
    except ValueError as e:
        print(f"Could not calculate AUC: {e}") # Handle cases like only one class in test preds/labels
        test_auc = 0.0 # Assign a default value or handle as needed
else:
     print("AUC calculation skipped (not binary or only one class found in test labels)")
     test_auc = 0.0 # Assign a default value

cm = confusion_matrix(test_all_labels, test_all_preds)
# Store results in dictionary for plotting
results = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'test_auc': test_auc if test_auc is not None else 0.0, # Ensure AUC is float for plot
    'confusion_matrix': cm
}

# Function to plot results
def plot_results(results, class_names, save_dir):
    plt.figure(figsize=(15, 10))

    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(results.get('train_losses', []), label='Training Loss')
    plt.plot(results.get('val_losses', []), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(results.get('val_accuracies', []), label='Validation Accuracy', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot confusion matrix if available
    if 'confusion_matrix' in results and results['confusion_matrix'] is not None:
        plt.subplot(2, 2, 3)
        cm = results['confusion_matrix']
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Test Set)')
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

    # Plot additional test metrics if available
    if 'test_accuracy' in results:
        plt.subplot(2, 2, 4)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        values = [
            results.get('test_accuracy', 0) / 100.0, # Convert accuracy to 0-1 scale
            results.get('test_precision', 0),
            results.get('test_recall', 0),
            results.get('test_f1', 0),
            results.get('test_auc', 0)
        ]
        bars = plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
        plt.title('Test Set Metrics')
        # Add text labels above bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom')


    plt.tight_layout()
    plot_filename = os.path.join(save_dir, 'training_results_2_0.png')
    plt.savefig(plot_filename)
    print(f"Results plot saved to {plot_filename}")
    plt.close()


# Plot the results using the adapted function
plot_results(results, class_names, CONFIG['save_dir'])