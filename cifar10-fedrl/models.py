import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from torchvision.models import resnet18,resnet34
import torch.nn as nn

class SimpleNN(nn.Module):
    """Simple neural network for MNIST-like datasets (from comprehensive simulation)."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ResNetFed(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetFed, self).__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        

    def forward(self, x):
        return self.model(x)
    
class SimpleMNISTCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # Output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x14x14
        self.pool = nn.MaxPool2d(2, 2)  # Downsamples by factor of 2
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 32x14x14
        x = self.pool(F.relu(self.conv2(x)))  # -> 64x7x7
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleCIFAR10CNN(nn.Module):
    """Simple 2-layer CNN for CIFAR-10 from FLASH-RL paper.
    
    NO BatchNorm - designed for stable federated learning with non-IID data.
    Architecture: Conv(5x5) -> ReLU -> MaxPool -> Conv(5x5) -> ReLU -> MaxPool -> FC -> FC
    ~466K parameters
    """
    def __init__(self, num_classes=10):
        super(SimpleCIFAR10CNN, self).__init__()
        
        # First convolutional layer: 3 -> 32 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional layer: 32 -> 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1600, 256)  # 64 * 5 * 5 = 1600 after both maxpools from 32x32
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)           # 3x32x32 -> 32x28x28
        x = F.relu(x)
        x = self.maxpool1(x)        # 32x28x28 -> 32x14x14
        
        # Second conv block
        x = self.conv2(x)           # 32x14x14 -> 64x10x10
        x = F.relu(x)
        x = self.maxpool2(x)        # 64x10x10 -> 64x5x5
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)     # 64x5x5 -> 1600
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()

        # Feature Extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input: 3x32x32 → 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16x16
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x4x4
            nn.Dropout(0.2),
        )

        # Classifier
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x