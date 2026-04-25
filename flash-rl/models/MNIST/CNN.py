import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_channels, out_channels1, out_channels2, num_classes):
        
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = out_channels1, kernel_size=5, stride = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv2 = nn.Conv2d(in_channels = out_channels1, out_channels = out_channels2, kernel_size=5, stride = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self,x):
        #first Convolutional layers
        x = self.conv1(x)
        #activation function 
        x = F.relu(x)
        #max pooling 
        x = self.maxpool1(x)
        #first Convolutional layers
        x = self.conv2(x)
        #activation function
        x = F.relu(x)
        #max pooling
        x = self.maxpool2(x)
        #flatten output 
        x = torch.flatten(x,1)
        #fully connected layer 1
        x =self.fc1(x)
        #activation function
        x = F.relu(x)
        #fully connected layer 2
        x = self.fc2(x)
        # get log probabilities
        #x = F.log_softmax(x, dim=1)
        return x


class SimpleMNISTCNN(nn.Module):
    """
    Simple CNN for MNIST from FedRL.
    
    Architecture:
    - Conv1: 1 -> 32 channels, 3x3 kernel, padding=1
    - MaxPool: 2x2
    - Conv2: 32 -> 64 channels, 3x3 kernel, padding=1
    - MaxPool: 2x2
    - Dropout: 0.25
    - FC1: 64*7*7 -> 128
    - FC2: 128 -> num_classes
    
    Input: 1x28x28 (MNIST images)
    Output: num_classes logits
    """
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