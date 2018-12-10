import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from my_cnn_v2 import Cnn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 4 * 64, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 4 * 4 * 64)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def initialize_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform


def load_cifar(transform):
    train_data = datasets.CIFAR10(
        'data', train=True, download=False, transform=transform)

    test_data = datasets.CIFAR10(
        'data', train=False, download=False, transform=transform)

    return train_data, test_data


def split_train(train, valid_size=0.2):
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx


def initialize_data_loader(data, batch_size, sampler=None, num_workers=0):
    return torch.utils.data.DataLoader(
        data, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


def train(model, epochs, train_loader, valid_loader, filename):
    valid_loss_min = np.Inf
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(1, epochs + 1):
        train_loss = 0
        valid_loss = 0

        # train model
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        # validate model
        model.eval()
        for data, target in valid_loader:
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.
              format(epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(
                'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.
                format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), '{}.pt'.format(filename))
            valid_loss_min = valid_loss


def test(model, filename, data_test, batch_size, classes):
    classes = classes
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(filename))
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()
    for data, target in data_test:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())

        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss / len(data_test.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' %
                  (classes[i], 100 * class_correct[i] / class_total[i],
                   np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' %
                  (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' %
          (100. * np.sum(class_correct) / np.sum(class_total),
           np.sum(class_correct), np.sum(class_total)))


if __name__ == '__main__':
    # prepare data
    batch_size = 20
    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ]
    transform = initialize_transform()
    train_data, test_data = load_cifar(transform)
    train_idx, valid_idx = split_train(train_data)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=0)

    valid_loader = initialize_data_loader(
        data=train_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(valid_idx),
        num_workers=0)

    test_loader = initialize_data_loader(
        data=test_data, batch_size=batch_size, num_workers=0)

    # # initialize model
    # model = Net()
    # model = Cnn()
    # print(model)
    # train model
    # train(
    #     model=model,
    #     epochs=30,
    #     train_loader=train_loader,
    #     valid_loader=valid_loader,
    #     filename='model1_cifar')

    # test(
    #     model=model,
    #     filename='model1_cifar.pt',
    #     data_test=test_loader,
    #     batch_size=batch_size,
    #     classes=classes)

    model1 = Net()
    model2 = Cnn()
    models = {'model_cifar.pt': model1, 'model1_cifar.pt': model2}
    for key, value in models.items():
        test(
            model=value,
            filename=key,
            data_test=test_loader,
            batch_size=batch_size,
            classes=classes)
