import torch
import torch.nn as nn
import torch.utils.data as data
from models import CNNModel, ResNetFed, SimpleMNISTCNN, SimpleNN


class Server:
    """Federated Learning Server."""
    def __init__(self, test_dataset, num_classes=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = ResNetFed().to(self.device)
        #self.global_model = SimpleNN().to(self.device)
        self.test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()

    def aggregate_models(self, client_models):
        """Aggregate client models using FedAvg."""
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([client_models[i][key].float() for i in range(len(client_models))], 0).mean(0)
        self.global_model.load_state_dict(global_dict)

    def evaluate(self):
        """Evaluate the global model on the test dataset."""
        self.global_model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        avg_loss = total_loss / len(self.test_loader)
        return accuracy, avg_loss
