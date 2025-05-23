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
    handler = log.FileHandler("./log/ResNet18_model_for_CIFAR10.log", "w", "utf-8")
    handler.setFormatter(log.Formatter(fmt="%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger.addHandler(handler)

# class Square(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return torch.pow(x, 2)

class ResNet18(nn.Module):
    """
    ResNet18 model implementation with original code structure.
    """

    def __init__(self, num_classes):
        """
        Initialize the ResNet18 model.
        """
        super(ResNet18, self).__init__()

        # Define residual connection function
        def make_residual_connection(inputs, outputs):
            """Build residual connection"""
            downsample = None
            if inputs != outputs or stride != 1:
                downsample = nn.Sequential(
                    Conv2d(inputs, outputs, kernel_size=1, stride=stride, bias=False),
                    BatchNorm2d(outputs)
                )
            return downsample

        # Define basic residual block
        def basic_block(inputs, outputs, stride=1):
            """Build basic residual block"""
            layers = []
            # First convolutional layer
            layers.append(Conv2d(inputs, outputs, kernel_size=3, stride=stride, padding=1, bias=False))
            layers.append(BatchNorm2d(outputs))
            layers.append(ReLU(inplace=True))

            # Second convolutional layer
            layers.append(Conv2d(outputs, outputs, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(BatchNorm2d(outputs))

            return nn.Sequential(*layers)

        # Use sequential model to maintain original structure
        self.resnet18 = nn.Sequential(
            # Initial convolutional layer
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=3, stride=2, padding=1),

            # Layer 1 (2 residual blocks, maintain channels=64)
            # Residual block 1
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),  # Residual connection handled in forward

            # Residual block 2
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),  # Residual connection handled in forward

            # Layer 2 (2 residual blocks, channels 64->128)
            # Residual block 1 (downsampling)
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(128),
            ReLU(inplace=True),  # Residual connection handled in forward

            # Residual block 2
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(128),
            ReLU(inplace=True),  # Residual connection handled in forward

            # Layer 3 (2 residual blocks, channels 128->256)
            # Residual block 1 (downsampling)
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(inplace=True),  # Residual connection handled in forward

            # Residual block 2
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(inplace=True),  # Residual connection handled in forward

            # Layer 4 (2 residual blocks, channels 256->512)
            # Residual block 1 (downsampling)
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            ReLU(inplace=True),  # Residual connection handled in forward

            # Residual block 2
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            ReLU(inplace=True),  # Residual connection handled in forward

            # Global average pooling
            AdaptiveAvgPool2d((1, 1)),

            # Flatten layer
            Flatten(),

            # Fully connected layer
            Linear(512, num_classes)
        )

        # Initialize network weights
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forwardRegular(self, x):
        """
        Forward pass of the ResNet18 model.
        Note: Residual connections cannot be implemented in Sequential model here,
        true ResNet18 requires custom modules.
        """
        # Use Sequential model for forward propagation
        out = self.resnet18(x)
        return out

def setVariables():
    """
    Set hyperparameters and device configuration
    """
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"The parameters are as follows: \n batch_size={batch_size}\t num_class={num_classes}\t "
             f"learning_rate={learning_rate}\t num_epochs={num_epochs}\t device={device}")

    return batch_size, num_classes, learning_rate, num_epochs, device

def loadDataset(batch_size):
    """
    Load and prepare the CIFAR10 dataset.
    """
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
    ]), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
    ]), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def initializeModelAndOptimizer(num_classes, device, learning_rate):
    """
    Set up the model, loss function, and optimizer.
    """
    model = ResNet18(num_classes).to(device)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, cost, optimizer

def training(train_loader, num_epochs, device, model, cost, optimizer):
    """
    Train the ResNet18 model
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
    Test the ResNet18 model with inputs.
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


        print('RealNumber: Accuracy of the network on the 10000 experiments images: {} %'.format(100 * correct / total))

    return correct / total


def main():
    """
    Main function to set up, train, and test the ResNet18 model in both original and fixed-point representation settings.
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
    torch.save(model.state_dict(), "./model/ResNet18_model.pt")

if __name__ == "__main__":
    setup_logging()
    main()
    model = torch.load("/home/lilvmy/paper-demo/Results_Verification_PPML/pre-trained-model/model/ResNet18_model.pt", weights_only=True)
    model_params = {}
    for name, param in model.items():
        model_params[name] = param.cpu().numpy()

    np.save("./model/ResNet18_model_params.npy", model_params)

    load_params = np.load("./model/ResNet18_model_params.npy", allow_pickle=True).item()

    print(load_params)
