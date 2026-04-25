import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from models import CNNModel, ResNetFed, SimpleMNISTCNN, SimpleNN, SimpleCIFAR10CNN


class Client:
    """Federated Learning Client."""
    def __init__(self, client_id, dataset, num_classes = 10):
        self.client_id = client_id
        self.local_data = data.DataLoader(dataset, batch_size=128, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = ResNetFed().to(self.device)
        self.model = ResNetFed().to(self.device)
        #self.model = SimpleNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)


    def train(self, epochs=5):
        """Train the client's model locally."""
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(epochs):
            runningLoss = 0.0
            total, correct = 0, 0
            for images, labels in self.local_data:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                runningLoss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            epoch_loss = runningLoss / len(self.local_data)
            epoch_accuracy = correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            
            # Print detailed training info for each client to show local metrics
            print(f"Client ID: {self.client_id}, Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            self.scheduler.step()

        # Return model state dict along with training metrics
        return {
            'model_state': self.model.state_dict(),
            'losses': epoch_losses,
            'accuracies': epoch_accuracies,
            'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
            'final_accuracy': epoch_accuracies[-1] if epoch_accuracies else 0.0
        }

    def get_class_distribution(self):
        """Get class distribution in client dataset."""
        # Extract labels from the dataset
        labels = []
        for _, label in self.local_data.dataset:
            labels.append(label)
        
        # Convert to tensor and compute class counts
        labels_tensor = torch.tensor(labels)
        class_counts = torch.bincount(labels_tensor, minlength=10)  # Ensure 10 classes
        
        # Return normalized distribution
        return class_counts.float() / class_counts.sum()
