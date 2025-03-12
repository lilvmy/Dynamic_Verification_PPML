import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import *
import torch.utils.data.dataloader as Dataloader
import numpy as np
import logging as log

torch.manual_seed(42)

def setup_logging():
    """
    Add log
    """
    root_logger = log.getLogger()
    root_logger.setLevel(log.INFO)
    handler = log.FileHandler("log/cnn2_model_for_mnist.log", "w", "utf-8")
    handler.setFormatter(log.Formatter(fmt="%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger.addHandler(handler)

class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(x, 2)


class CNN2(nn.Module):
    """
    CNN2 model implementation.
    """

    def __init__(self, num_classes):
        """
        Initialize the CNN2 model.
        """
        super(CNN2, self).__init__()
        self.cnn2 = nn.Sequential(
            Conv2d(1, 64, kernel_size=3),
            Square(),
            AvgPool2d(kernel_size=4, stride=2),
            Flatten(),
            Linear(64 * 12 * 12, 32),
            Linear(32, num_classes)
        )


    def forwardRegular(self, x):
        """
        Forward pass of the regular cnn2 model.
        """
        out = self.cnn2(x)
        return out

def setVariables():
    """
    Set hyperparameters and device configuration
    """
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"The parameters are as follows: \n batch_size={batch_size}\t num_class={num_classes}\t "
             f"learning_rate={learning_rate}\t num_epochs={num_epochs}\t device={device}")

    return batch_size, num_classes, learning_rate, num_epochs, device

def loadDataset(batch_size):
    """
    Load and prepare the MNIST dataset.
    """
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),
                                              download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def initializeModelAndOptimizer(num_classes, device, learning_rate):
    """
    Set up the model, loss function, and optimizer.
    """
    model = CNN2(num_classes).to(device)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, cost, optimizer

def training(train_loader, num_epochs, device, model, cost, optimizer):
    """
    Train the CNN2 model.
    """
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model.forwardRegular(images)
            loss = cost(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 400 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                log.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}")

def testing(test_loader, device, model):
    """
    Test the CNN2 model with inputs.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forwardRegular(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        print('RealNumber: Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    return correct / total


def main():
    """
    Main function to set up, train, and test the CNN2 model in both original and fixed-point representation settings.
    """
    batch_size, num_classes, learning_rate, num_epochs, device = setVariables()

    train_loader, test_loader = loadDataset(batch_size)
    model, cost, optimizer = initializeModelAndOptimizer(num_classes, device, learning_rate)
    log.info(f"The detail model description is {model}")

    print("\nTraining...")
    log.info("Training.........")
    training(train_loader, num_epochs, device, model, cost, optimizer)

    print("\nTesting...")
    log.info("Testing.........")
    acc = testing(test_loader, device, model)
    log.info(f"The accuracy of real number model is {acc}")

    print("\nSave Model...")
    torch.save(model.state_dict(), "./model/cnn2_model.pt")

if __name__ == "__main__":
    setup_logging()
    main()



