import torch
import torch.nn as nn
import torch.utils.data as data


def _jain_fairness_index(accs):
    """Jain's Fairness Index: (Σaᵢ)² / (N·Σaᵢ²). Returns 1.0 for a single client."""
    n = len(accs)
    if n == 0:
        return float("nan")
    s1 = sum(accs)
    s2 = sum(a * a for a in accs)
    if s2 == 0.0:
        return 1.0
    return (s1 ** 2) / (n * s2)


class Server:
    """Federated Learning Server.

    The global model must be set externally before calling evaluate_per_client()
    or aggregate_models():
        server.global_model = MyModel().to(server.device)
    """

    def __init__(self, client_test_datasets, num_classes=10):
        """
        Args:
            client_test_datasets: list of torch Dataset objects, one per client.
                                  Used for per-client fairness evaluation.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = None   # Set from main after construction
        self.criterion = nn.CrossEntropyLoss()
        self._client_test_loaders = [
            data.DataLoader(ds, batch_size=128, shuffle=False)
            for ds in client_test_datasets
        ]

    def aggregate_models(self, client_models):
        """Aggregate client models using FedAvg."""
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack(
                [client_models[i][key].float() for i in range(len(client_models))], 0
            ).mean(0)
        self.global_model.load_state_dict(global_dict)

    def evaluate_per_client(self):
        """Evaluate the global model on each client's local test set.

        Primary metrics (equal weight per client, no size bias):
            mean_acc  (float) – mean per-client accuracy
            std_acc   (float) – std deviation of per-client accuracies
            jfi       (float) – Jain's Fairness Index in (0,1], higher = fairer

        Returns:
            mean_acc, std_acc, jfi, per_client_accs (list[float]), avg_loss (float)
        """
        self.global_model.eval()
        per_client_accs = []
        per_client_losses = []

        with torch.no_grad():
            for loader in self._client_test_loaders:
                if len(loader) == 0:
                    per_client_accs.append(0.0)
                    per_client_losses.append(0.0)
                    continue
                correct, total, total_loss = 0, 0, 0.0
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.global_model(images)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                per_client_accs.append(correct / total if total > 0 else 0.0)
                per_client_losses.append(total_loss / len(loader))

        n = len(per_client_accs)
        mean_acc = sum(per_client_accs) / n if n > 0 else 0.0
        variance = sum((a - mean_acc) ** 2 for a in per_client_accs) / n if n > 0 else 0.0
        std_acc = variance ** 0.5
        jfi = _jain_fairness_index(per_client_accs)
        avg_loss = sum(per_client_losses) / len(per_client_losses) if per_client_losses else 0.0

        return mean_acc, std_acc, jfi, per_client_accs, avg_loss
