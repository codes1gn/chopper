import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np

from chopper.pytorch import *


# jif torch.cuda.is_available():
#    device = torch.device("cuda")
# else:
#    device = torch.device("cpu")
device = torch.device("cpu")

print("Using PyTorch version:", torch.__version__, " Device:", device)
batch_size = 32

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
validation_dataset = datasets.MNIST("./data", train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

"""
    @backend("IREE")
    @annotate_arguments(
        [
            None,
            (x_shape, torch.float32),
            (kernel_shape, torch.float32),
        ]
    )

"""


class MLP_Layer1(nn.Module):
    def __init__(self):
        super().__init__()
        # self.x_shape = x_shape
        # self.kernel_shape = kernel_shape
        # self.forward = backend("IREE")(
        #     annotate_arguments([None, (x_shape, torch.float32), (kernel_shape, torch.float32)])(self.forward)
        # )

    @backend("IREE")
    @annotate_arguments(
        [
            None,
            ([32, 784], torch.float32),
            ([784, 50], torch.float32),
        ]
    )
    def forward(self, x, x1):
        y = torch.matmul(x, x1)
        return y


class MLP_Layer2(nn.Module):
    def __init__(self):
        super().__init__()
        # self.x_shape = x_shape
        # self.kernel_shape = kernel_shape
        # self.forward = backend("IREE")(
        #     annotate_arguments([None, (x_shape, torch.float32), (kernel_shape, torch.float32)])(self.forward)
        # )

    @backend("IREE")
    @annotate_arguments(
        [
            None,
            ([32, 50], torch.float32),
            ([50, 50], torch.float32),
        ]
    )
    def forward(self, x, x1):
        y = torch.matmul(x, x1)
        return y


class MLP_Layer3(nn.Module):
    def __init__(self):
        super().__init__()
        # self.x_shape = x_shape
        # self.kernel_shape = kernel_shape
        # self.forward = backend("IREE")(
        #     annotate_arguments([None, (x_shape, torch.float32), (kernel_shape, torch.float32)])(self.forward)
        # )

    @backend("IREE")
    @annotate_arguments(
        [
            None,
            ([32, 50], torch.float32),
            ([50, 10], torch.float32),
        ]
    )
    def forward(self, x, x1):
        y = torch.matmul(x, x1)
        return y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.kernel1 = nn.Parameter(
            Variable(torch.empty((28 * 28, 50), dtype=torch.float32).normal_(0.0, 0.05), requires_grad=True)
        )
        self.mlp_layer1 = MLP_Layer1()
        self.kernel2 = nn.Parameter(torch.empty((50, 50), dtype=torch.float32).normal_(0.0, 0.05).requires_grad_(True))
        self.mlp_layer2 = MLP_Layer2()
        self.kernel3 = nn.Parameter(torch.empty((50, 10), dtype=torch.float32).normal_(0.0, 0.05).requires_grad_(True))
        self.mlp_layer3 = MLP_Layer3()

    def forward(self, x):
        x = self.mlp_layer1(x, self.kernel1)
        x = self.mlp_layer2(x, self.kernel2)
        x = self.mlp_layer3(x, self.kernel3)
        return F.log_softmax(x, dim=1)


model = Net().to(device)
print(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)


def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.view(-1, 28 * 28)
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )


def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            val_loss, correct, len(validation_loader.dataset), accuracy
        )
    )


epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)
