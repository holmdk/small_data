import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import MLP

#
# MNIST consists of 60k training and 10k test examples from
# 10 classes (LeCun, 1998). We train MLPs of various depth
# and width, with and without dropout, as well as standard
# ConvNets on this dataset. Unless otherwise noted, we use
# ReLU as the nonlinearity.



" OPTIMIZERS "
# Adam (Kingma & Ba, 2014) with fixed learning rates
# {10−4 , 3 · 10−4,  10−3} and 50 epochs.


# Momentum SGD with initial learning rates {10−4, 3 · 10−4, .., 10−1} cosine-decaying over 50 epochs down to
#0 (0.9 momentum and epsilon = 10−4 )


# For all experiments we use a batch size of 256 examples.
# The term epoch always refers to the number of gradient steps
# required to go through the full-sized dataset once; i.e., on
# ImageNet an epoch is always 1.28M/256 = 5000 gradient
# steps, regardless of the size of the actual training set used


#################

" PARAMETERS "


BATCH_SIZE = 256
print_every_n = 200
val_n = 10000


model = MLP()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#################




kwargs = {'batch_size': BATCH_SIZE}

torch.manual_seed(1234)

device = torch.device("cuda")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))    #transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True,
                          transform=transform)

train_set, val_set = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_n, val_n])

test_dataset = datasets.MNIST('../data', train=False,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
valid_loader = torch.utils.data.DataLoader(val_set, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)



# resnet18 = models.resnet18()


loss_fn = nn.CrossEntropyLoss()

mean_train_losses = []
mean_valid_losses = []
train_losses = []
valid_losses = []
valid_acc_list = []

for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):

        # temperature scaling
        


        # get the inputs; data is a list of [inputs, labels]
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # print statistics
        running_loss += loss.item()
        if i % print_every_n == (print_every_n-1):  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / print_every_n))
            running_loss = 0.0

    model.eval()
    correct = 0
    total = 0
    valid_losses = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            valid_losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))

    accuracy = 100 * correct / total
    valid_acc_list.append(accuracy)
    print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%' \
          .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses), accuracy))

print('Finished Training')




correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))






