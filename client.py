# client.py
import flwr as fl
import torch
from collections import OrderedDict
from model import DiabetesNet
from utils import load_data

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
NUM_CLIENTS = 2
CLIENT_ID = int(input("Enter client ID (0 or 1): "))

print(f"\nClient {CLIENT_ID} connected to server")

# Initialize model and data
model = DiabetesNet().to(DEVICE)
trainloader, testloader = load_data(CLIENT_ID, NUM_CLIENTS)

# Get model parameters
def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# Set model parameters
def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Local training function (UPDATED)
def train(net, trainloader, epochs=5):
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    print("Training local model...")

    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in trainloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = net(X).view(-1)
            loss = loss_fn(output, y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print only first and last epoch (clean output)
        if epoch == 0 or epoch == epochs - 1:
            avg_loss = total_loss / len(trainloader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.2f}")

# Evaluation function
def test(net, testloader):
    net.eval()
    correct, total, loss = 0, 0, 0.0
    loss_fn = torch.nn.BCELoss()

    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = net(X).view(-1)
            preds = outputs > 0.5
            correct += (preds == y.view(-1)).sum().item()
            loss += loss_fn(outputs, y.view(-1)).item()
            total += y.size(0)

    return loss / total, correct / total

# Flower client
class DiabetesClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(model)

    def fit(self, parameters, config):
        set_parameters(model, parameters)
        train(model, trainloader, epochs=5)
        print("Sending model update to server\n")
        return get_parameters(model), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(model, parameters)
        loss, accuracy = test(model, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Start client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=DiabetesClient()
)
