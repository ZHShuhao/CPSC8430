import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def data_loader(train_batch_size, test_batch_size):
    train_data = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor()
                       ])),
        batch_size=train_batch_size, shuffle=True)

    test_data = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])),
        batch_size=test_batch_size, shuffle=True)

    return (train_data, test_data)



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
def sensitivity(model):
    fNormTotal = 0
    counter = 0
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = p.grad
            fNorm = torch.linalg.norm(grad).item()  # Use .item() to get a Python float
            fNormTotal += fNorm
            counter += 1
    return fNormTotal / counter

def init_optimizer(model):
    return optim.SGD(model.parameters(), lr=1e-2)


def train(model, optimizer, data):
    model.train()
    for batch_idx, (data, target) in enumerate(data):
        data, target = data.to(device), target.to(device)  # Move data to GPU
        optimizer.zero_grad()
        output = model(data)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


def calculate_op_diff(model, loader):
    correct = 0
    total = 0
    costTotal = 0
    costCounter = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            data, target = batch
            data, target = data.to(device), target.to(device)  # Move data to GPU
            output = model(data)
            cost = loss_fn(output, target)
            costTotal += cost.item()  # Convert tensor to float
            costCounter += 1
            for i, outputTensor in enumerate(output):
                if torch.argmax(outputTensor) == target[i]:
                    correct += 1
                total += 1
    return costTotal / costCounter, round(correct / total, 3)



# Load checkpoint
def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        test_loss = checkpoint['test_loss']
        train_acc = checkpoint['train_acc']
        test_acc = checkpoint['test_acc']
        return model, optimizer, epoch, train_loss, test_loss, train_acc, test_acc
    else:
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

# Save checkpoint
def save_checkpoint(epoch, model, optimizer, train_loss, test_loss, train_acc, test_acc, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_acc': train_acc,
        'test_acc': test_acc
    }
    torch.save(checkpoint, filename)


epochs = 50
param_arr = []
train_loss_arr = []
test_loss_arr = []
train_accuracy_arr = []
test_accuracy_arr = []

# Function to train models, saving loss and accuracy for each epoch
def train_models(model, optimizer, train_data, test_data, start_epoch=1):
    model.to(device)  # Move model to GPU
    for epoch in range(start_epoch, epochs + 1):
        print(f"Training epoch - {epoch}")
        train(model, optimizer, train_data)
        train_loss, train_acc = calculate_op_diff(model, train_data)
        test_loss, test_acc = calculate_op_diff(model, test_data)
        train_loss_arr.append(train_loss)
        test_loss_arr.append(test_loss)
        train_accuracy_arr.append(train_acc)
        test_accuracy_arr.append(test_acc)
        print(f"Epoch {epoch} | Train Loss: {train_loss}, Test Loss: {test_loss}, Train Acc: {train_acc}, Test Acc: {test_acc}")
        save_checkpoint(epoch, model, optimizer, train_loss, test_loss, train_acc, test_acc)
    print('Training completed for all epochs.')

batch_arr = [150, 250, 750, 4000, 8000]
sensitivity_arr = []

# Train models using different batch sizes
for i, batch in enumerate(batch_arr, start=1):
    print(f"Training model {i} with batch size: {batch}")
    train_data, test_data = data_loader(batch, batch)
    model = Model()
    optimizer = init_optimizer(model)
    try:
        model, optimizer, start_epoch, train_loss, test_loss, train_acc, test_acc = load_checkpoint(model, optimizer, filename=f"checkpoint_model_{i}.pth")
        print(f"Resuming model {i} from epoch {start_epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Acc: {train_acc}, Test Acc: {test_acc}")
    except FileNotFoundError:
        print(f"No checkpoint found for model {i}, starting from scratch.")
        start_epoch = 1
    train_models(model, optimizer, train_data, test_data, start_epoch=start_epoch)
    sensitivity_arr.append(sensitivity(model))
    print(f"Model {i} training completed with batch size: {batch}")


# Plotting the results
import matplotlib.pyplot as plt


min_length = min(len(batch_arr), len(train_loss_arr), len(test_loss_arr), len(sensitivity_arr))


plt.figure(figsize=(12,6))


plt.plot(batch_arr[:min_length], train_loss_arr[:min_length], color="red", label='Train Loss')
plt.plot(batch_arr[:min_length], test_loss_arr[:min_length], color="purple", label='Test Loss')

plt.xlabel('Batch size')
plt.ylabel('Loss')
plt.legend(loc='upper left')

ax2 = plt.gca().twinx()
ax2.plot(batch_arr[:min_length], sensitivity_arr[:min_length], color="grey", label='Sensitivity')
ax2.set_ylabel('Sensitivity')

plt.title('Model Loss Comparison')
plt.legend(['Sensitivity'], loc='upper right')
plt.savefig('par2-loss')


# second graph
min_length = min(len(batch_arr), len(train_accuracy_arr), len(test_accuracy_arr), len(sensitivity_arr))


plt.figure(figsize=(12,6))
plt.plot(batch_arr[:min_length], train_accuracy_arr[:min_length], color="orange", label='Train Accuracy')
plt.plot(batch_arr[:min_length], test_accuracy_arr[:min_length], color="purple", label='Test Accuracy')

plt.xlabel('Batch size')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')


ax2 = plt.gca().twinx()
ax2.plot(batch_arr[:min_length], sensitivity_arr[:min_length], color="grey", label='Sensitivity')
ax2.set_ylabel('Sensitivity')

plt.title('Model Accuracy Comparison')
plt.legend(['Sensitivity'], loc='upper right')
plt.savefig('par2-accuracy')






















