import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split # Added random_split here
from tqdm import tqdm
import multiprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import torch.nn.functional as F # <-- Added F for softmax
import numpy as np            # <-- Added numpy
import os                     # <-- Added os

# --- Configuration ---
# Set the path to the dataset directory
# PLEASE UPDATE THIS PATH TO YOUR ACTUAL DATASET LOCATION
data_dir = "/home/rikisu/NNDL/CNN/cell_images" # <--- MAKE SURE THIS PATH IS CORRECT

# Set the number of training epochs
epochs = 40 # Increase for potentially better results

# Automatically set the number of worker processes for data loading
num_workers = multiprocessing.cpu_count()
print(f"Using {num_workers} workers for data loading.")

# Directory to save results
save_dir = 'results'
os.makedirs(save_dir, exist_ok=True) # Create save directory if it doesn't exist

# --- Data Transformations ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizing helps training
])

# --- Load and Split Dataset ---
try:
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes # Get class names
    print(f"Dataset loaded successfully. Found {len(full_dataset)} images in {len(class_names)} classes: {class_names}")
except FileNotFoundError:
    print(f"ERROR: Dataset directory not found at '{data_dir}'. Please ensure the path is correct.")
    exit() # Exit if dataset path is wrong
except Exception as e:
    print(f"An error occurred loading the dataset: {e}")
    exit()

# Define split sizes (e.g., 80% train, 10% validation, 10% test)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

print(f"Splitting dataset into: Train={train_size}, Validation={val_size}, Test={test_size}")

# Split the dataset
# Use a fixed generator for reproducibility if desired
# generator = torch.Generator().manual_seed(42)
# train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])


# --- Create Data Loaders ---
batch_size = 16 # You can adjust this based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# --- Define CNN Model ---
class MalariaCNN(nn.Module):
    def __init__(self, num_classes=2): # Pass num_classes
        super(MalariaCNN, self).__init__()
        #---------- CONVOLUTIONAL LAYERS ----------#
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 128 -> 64

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # pool: 64 -> 32

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        # pool: 32 -> 16

        # Adaptive pooling can make the FC layer input size independent of input image size changes
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6)) # Example: Output size 6x6
        # self.fc1 = nn.Linear(128 * 6 * 6, 128)

        # Calculate the flattened size dynamically or ensure it matches the architecture
        # Input: (batch_size, 3, 128, 128)
        # After conv1+pool: (batch_size, 32, 64, 64)
        # After conv2+pool: (batch_size, 64, 32, 32)
        # After conv3+pool: (batch_size, 128, 16, 16)
        self.flattened_size = 128 * 16 * 16
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Added dropout for regularization
        self.fc2 = nn.Linear(128, num_classes) # Output layer size based on num_classes

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = self.pool(self.relu3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        # x = x.view(-1, self.flattened_size) # Alternative flattening

        x = self.relu4(self.fc1(x))
        x = self.dropout(x) # Apply dropout
        x = self.fc2(x)
        return x

# --- Model Setup ---
# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MalariaCNN(num_classes=len(class_names)).to(device) # Pass number of classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Enable mixed precision training if GPU is available
use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)# Correct scaler initialization

print("\n--- Starting Training ---")
train_losses = []
val_losses = [] # <-- Added validation loss list
val_accuracies = [] # <-- Renamed from test_accuracies

for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision context
        with torch.amp.autocast(device_type='cuda', enabled=use_amp): 
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        # Show batch loss in tqdm
        progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")

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

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_running_loss / len(val_loader)
    val_losses.append(avg_val_loss) # <-- Store validation loss
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy) # <-- Store validation accuracy

    print(f"Epoch {epoch+1}/{epochs} => Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

print("\n--- Training Finished ---")


# --- Final Test Set Evaluation ---
print("\n--- Evaluating on Test Set ---")
model.eval()
test_correct = 0
test_total = 0
test_all_preds = []
test_all_labels = []
test_all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="[Test Eval]"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp): 
            outputs = model(images)

        probs = F.softmax(outputs, dim=1) # Get probabilities
        _, predicted = torch.max(outputs.data, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        test_all_preds.extend(predicted.cpu().numpy())
        test_all_labels.extend(labels.cpu().numpy())
        # Store probability of the positive class (assuming binary, class index 1)
        if len(class_names) == 2:
             test_all_probs.extend(probs[:, 1].cpu().numpy())
        # Add multi-class probability handling here if needed

# --- Calculate Final Metrics ---
# Moved calculation after the loop
test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0

# Use average='binary' for binary classification. Adjust if multi-class.
# Use 'weighted' or 'macro' for multi-class if needed
avg_method = 'binary' if len(class_names) == 2 else 'weighted'

test_precision = precision_score(test_all_labels, test_all_preds, average=avg_method, zero_division=0)
test_recall = recall_score(test_all_labels, test_all_preds, average=avg_method, zero_division=0)
test_f1 = f1_score(test_all_labels, test_all_preds, average=avg_method, zero_division=0)

test_auc = None
# Calculate AUC only for binary classification and if both classes are present in labels
if len(class_names) == 2 and len(np.unique(test_all_labels)) > 1 and len(test_all_probs) > 0:
    try:
        test_auc = roc_auc_score(test_all_labels, test_all_probs)
        print(f"Test AUC: {test_auc:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC: {e}. Assigning AUC = 0.0")
        test_auc = 0.0 # Assign a default value
else:
     print("AUC calculation skipped (not binary, only one class found, or probabilities missing). Assigning AUC = 0.0")
     test_auc = 0.0 # Assign a default value

print(f"\n--- Test Set Results ---")
print(f"Accuracy:  {test_accuracy:.2f}%")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
if test_auc is not None: # Check if AUC was calculated
    print(f"AUC:       {test_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(test_all_labels, test_all_preds)
print("\nConfusion Matrix:")
print(cm)


# --- Store Results for Plotting ---
# Ensure all metrics are calculated before this point
results = {
    'train_losses': train_losses,
    'val_losses': val_losses,         # <-- Use validation losses
    'val_accuracies': val_accuracies, # <-- Use validation accuracies
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'test_auc': test_auc if test_auc is not None else 0.0, # Ensure AUC is float
    'confusion_matrix': cm
}

# --- Visualization ---
def plot_results(results_dict, class_names_list, save_path):
    plt.figure(figsize=(15, 10))

    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(results_dict.get('train_losses', []), label='Training Loss', marker='o')
    plt.plot(results_dict.get('val_losses', []), label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(results_dict.get('val_accuracies', []), label='Validation Accuracy', color='g', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)


    # Plot confusion matrix
    cm_plot = results_dict.get('confusion_matrix')
    if cm_plot is not None and len(class_names_list) > 0:
        plt.subplot(2, 2, 3)
        plt.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Test Set)')
        plt.colorbar()
        tick_marks = np.arange(len(class_names_list))
        plt.xticks(tick_marks, class_names_list, rotation=45)
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
        plt.tight_layout() # Adjust layout to prevent labels overlapping plot


    # Plot additional test metrics bar chart
    plt.subplot(2, 2, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    values = [
        results_dict.get('test_accuracy', 0) / 100.0, # Scale accuracy to 0-1
        results_dict.get('test_precision', 0),
        results_dict.get('test_recall', 0),
        results_dict.get('test_f1', 0),
        results_dict.get('test_auc', 0) # Already 0-1
    ]
    bars = plt.bar(metrics, values, color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']) # Use distinct colors
    plt.title('Final Test Set Metrics')
    plt.ylabel('Score')
    plt.ylim(0.9, 1.1) # Set y-limit slightly above 1.0
    # Add text labels above bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save the plot
    plot_filename = os.path.join(save_path, 'training_validation_test_results_Team5.png')
    try:
        plt.savefig(plot_filename)
        print(f"\nResults plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close() # Display the plot

# Call the plotting function
plot_results(results, class_names, save_dir)

print("\n--- Script Finished ---")