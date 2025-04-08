import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

# Set the path to the dataset directory
data_dir = "/home/rikisu/NNDL/cell_images"

# Set the number of training epochs (you can increase this later for better accuracy)
epochs = 2

# Automatically set the number of worker processes for data loading
# based on the number of available CPU cores
num_workers = multiprocessing.cpu_count()

# Define CNN model
class MalariaCNN(nn.Module):
    def __init__(self):
        super(MalariaCNN, self).__init__()
        
        #---------- CONVOLUTIONAL LAYERS ----------#
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: 3 channels (RGB), Output: 32 feature maps
        self.relu1 = nn.ReLU()  # Activation function
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Reduces spatial dimensions by half
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Input: 32 channels, Output: 64 feature maps
        self.relu2 = nn.ReLU()  # Activation function
        # MaxPool reused from above
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Input: 64 channels, Output: 128 feature maps
        self.relu3 = nn.ReLU()  # Activation function
        # MaxPool reused from above
        
        #---------- FULLY CONNECTED LAYERS ----------#
        # First fully connected layer
        # After 3 max pooling operations (each reducing dimensions by 2), the 200x200 image becomes 25x25
        # With 128 feature maps, we get 128*25*25 input features
        self.fc1 = nn.Linear(128 * 25 * 25, 128)  # Dense layer to reduce dimensionality
        self.relu4 = nn.ReLU()  # Activation function
        
        # Output layer
        self.fc2 = nn.Linear(128, 2)  # Binary classification (parasitized vs uninfected)

    def forward(self, x):
        # First convolutional block with pooling
        x = self.conv1(x)        # Apply convolution
        x = self.relu1(x)        # Apply activation
        x = self.pool(x)         # Apply max pooling (200x200 -> 100x100)
        
        # Second convolutional block with pooling
        x = self.conv2(x)        # Apply convolution
        x = self.relu2(x)        # Apply activation
        x = self.pool(x)         # Apply max pooling (100x100 -> 50x50)
        
        # Third convolutional block with pooling
        x = self.conv3(x)        # Apply convolution
        x = self.relu3(x)        # Apply activation
        x = self.pool(x)         # Apply max pooling (50x50 -> 25x25)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Reshape to [batch_size, 128*25*25]
        
        # Fully connected layers
        x = self.fc1(x)          # First dense layer
        x = self.relu4(x)        # Apply activation
        x = self.fc2(x)          # Output layer (logits)
        
        return x

#---------- DATA PREPARATION ----------#
# Define data transformations
transform = transforms.Compose([
    transforms.Resize((200, 200)),  # Resize all images to 200x200
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
])

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)  # Expects data in folders named by class
train_size = int(0.8 * len(dataset))  # 80% training data
test_size = len(dataset) - train_size  # 20% testing data
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

#---------- MODEL SETUP ----------#
# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = MalariaCNN().to(device)  # Move model to GPU if available
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

#---------- TRAINING LOOP ----------#
train_losses = []
test_accuracies = []

# Enable mixed precision training
scaler = torch.amp.GradScaler()  # Fixed: removed 'cuda' parameter

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    for images, labels in progress_bar:
        # Move data to device
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()  # Clear gradients
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):  # Mixed precision
            outputs = model(images)  # Get model predictions
            loss = criterion(outputs, labels)  # Calculate loss

        # Backward pass with gradient scaling for mixed precision
        scaler.scale(loss).backward()  # Compute gradients
        scaler.step(optimizer)  # Update weights
        scaler.update()  # Update scaler

        # Track loss
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

    # Calculate average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    #---------- EVALUATION ----------#
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)  # Get model predictions
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Calculate and store accuracy
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%")

#---------- VISUALIZATION ----------#
# Plot results
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', color='b')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss over Epochs")

# Plot test accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies, marker='o', linestyle='-', color='r')
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy over Epochs")

plt.tight_layout()
plt.show()
