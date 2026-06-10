import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F


class Client:
    """Federated Learning Client.

    The model must be set externally before calling train():
        client.model = MyModel().to(client.device)
        client.optimizer = torch.optim.Adam(client.model.parameters(), lr=0.001)
    """

    def __init__(self, client_id, dataset, num_classes=10):
        self.client_id = client_id
        self.local_data = data.DataLoader(dataset, batch_size=128, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None       # Set from main after construction
        self.optimizer = None   # Set from main after construction
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = None   # Optionally set from main

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

            print(f"Client ID: {self.client_id}, Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

        return {
            'model_state': self.model.state_dict(),
            'losses': epoch_losses,
            'accuracies': epoch_accuracies,
            'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
            'final_accuracy': epoch_accuracies[-1] if epoch_accuracies else 0.0,
        }

    def get_class_distribution(self):
        """Get class distribution in client dataset."""
        labels = []
        for _, label in self.local_data.dataset:
            labels.append(label)
        labels_tensor = torch.tensor(labels)
        class_counts = torch.bincount(labels_tensor, minlength=10)
        return class_counts.float() / class_counts.sum()
