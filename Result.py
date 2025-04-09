import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import sys

# Define the correct model architecture based on error messages
class MalariaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MalariaCNN, self).__init__()
        
        # Initial convolution layer with 32 filters (not 16)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Additional convolution layers (instead of residual blocks)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Feature size calculation for the original model
        # After 3 pooling layers: 128/8 = 16
        feature_size = 128 * 16 * 16  # 32768
        
        # Fully connected layers with correct dimensions
        self.fc1 = nn.Linear(feature_size, 256)  # Not 128
        self.bn_fc1 = nn.BatchNorm1d(256)  # Not 128
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, num_classes)  # From 256 to num_classes
        
    def forward(self, x):
        # First convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 128x128 -> 64x64
        
        # Second convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64x64 -> 32x32
        
        # Third convolution block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 32x32 -> 16x16
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        
        return x

def load_model(model_path, device):
    """Load the pre-trained model"""
    model = MalariaCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_image(image_path, image_size=128):
    """Process the input image for the model"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Open image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict(model, image_tensor, device):
    """Make prediction on the image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.max(outputs, 1)[1].item()
        confidence = probabilities[0][predicted_class].item() * 100
    
    return predicted_class, confidence

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Malaria Cell Detection')
    parser.add_argument('--model', type=str, default='/home/rikisu/NNDL/CNN/results_improved_Team5/best_model_Team5.pth',
                        help='Path to the pre-trained model')
    parser.add_argument('--image', type=str, help='Path to the image for prediction')
    args = parser.parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(args.model, device)
        print(f"Model loaded successfully from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # If image path is not provided, ask for it
    image_path = args.image
    while not image_path or not os.path.exists(image_path):
        image_path = input("Enter the path to the cell image: ").strip()
        if not image_path:
            print("Please provide a valid image path.")
        elif not os.path.exists(image_path):
            print(f"Image not found at {image_path}")
            image_path = None
    
    # Process image
    image_tensor = process_image(image_path)
    if image_tensor is None:
        return
    
    # Make prediction
    # Swapped class ordering based on feedback
    class_names = ['Parasitized', 'Uninfected']
    predicted_class, confidence = predict(model, image_tensor, device)
    print(f"\nResult: {class_names[predicted_class]} (Confidence: {confidence:.2f}%)")
    # Additional information about the prediction
    if predicted_class == 0:  # Changed from 1 to 0 for Parasitized
        print("\nThe cell appears to be infected with malaria parasites.")
    else:
        print("\nThe cell appears to be uninfected.")
    print("\nNote: This tool is for demonstration purposes only and should not be used for clinical diagnosis.")
if __name__ == "__main__":
    main()