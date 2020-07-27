import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import MLP, ModelWithTemperature

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
# 0 (0.9 momentum and epsilon = 10−4 )


# For all experiments we use a batch size of 256 examples.
# The term epoch always refers to the number of gradient steps
# required to go through the full-sized dataset once; i.e., on
# ImageNet an epoch is always 1.28M/256 = 5000 gradient
# steps, regardless of the size of the actual training set used



# 30 different seeds



#################
" PARAMETERS "
include_temperature_scaling = False
BATCH_SIZE = 256
print_every_n = 200
val_n = 6000

lr = 0.0001
epochs = 50

SEEDS = range(1, 30)




#################

# class TempScaler(nn.Module):
#     def __init__(self):
#         super(TempScaler, self).__init__()
#         self.temperature = nn.Parameter(torch.ones(1) * 1.5, requires_grad=True).cuda()
#
#
#     def forward(self, logits):
#         return self.temperature_scale(logits)
#
#     def temperature_scale(self, logits):
#         """
#         Perform temperature scaling on logits
#         """
#         # Expand temperature to match the size of logits
#         temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
#         return logits / temperature
#
#     def set_temperature(self, logits, labels):
#         # Next: optimize the temperature w.r.t. NLL
#
#
#
#         def eval():
#             loss = nll_criterion(self.forward(logits), labels)
#             loss.backward()
#             return loss
#
#             # Next: optimize the temperature w.r.t. NLL
#         self.optimizer.step(eval)
#
#         return self


kwargs = {'batch_size': BATCH_SIZE}

torch.manual_seed(1234)

# tempscaler = TempScaler()

device = torch.device("cuda")


" DATA SETTINGS "

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=transform)

#TODO: Functionality for only selecting small proportion of training data

train_set, val_set = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_n, val_n])

test_dataset = datasets.MNIST('../data', train=False,
                              transform=transform)
batch_numbers = len(train_set) // BATCH_SIZE + 1

train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
valid_loader = torch.utils.data.DataLoader(val_set, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)





" MODEL SETTINGS "

model = MLP().to(device)
if include_temperature_scaling:
    model = ModelWithTemperature(model, valid_loader)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 10−3

# resnet18 = models.resnet18()

loss_fn = nn.CrossEntropyLoss().to(device)
# nll_criterion = nn.CrossEntropyLoss().cuda()

mean_train_losses = []
mean_test_losses = []

valid_acc_list = []
test_acc_list = []
test_entropy_list = []

for epoch in range(epochs):  # loop over the dataset multiple times

    model.train()

    train_losses = []
    test_losses = []

    for i, (images, labels) in enumerate(train_loader):

        ##################
        # temperature scaling
        ##################
        # if include_temperature_scaling:
        #
        #     logits_list = []
        #     labels_list = []
        #     with torch.no_grad:
        #         for _, (val_images, val_labels) in enumerate(valid_loader):
        #             outputs = model(val_images)
        #
        #             logits_list.append(outputs)
        #             labels_list.append(val_labels)
        #
        #         # optimizer.zero_grad()
        #
        #         logits = torch.cat(logits_list).cuda()
        #         val_labels = torch.cat(labels_list).cuda()
        #
        #     # before_temperature_nll = loss_fn(logits, labels).item()
        #
        #     # loss = loss_fn(temperature_scale(logits), val_labels)
        #
        #     tempscaler.set_temperature(logits, val_labels)
        #     #TODO: EVALUATE THAT THIS ACTUALLY DOES WHAT WE THINK IT DOES!
        #
        #     # valid_losses.append(loss.item())
        #
        #     # # print statistics
        #     # print('[EPOCH %d / %i, BATCH %5d / %i] generalized calibrated entropy: %.3f' %
        #     #       (epoch + 1, epochs, i + 1, batch_numbers, loss.item()))

        ##################
        # REGULAR TRAINING
        ##################

        # get the inputs; data is a list of [inputs, labels]
        # zero the parameter gradients


        # forward + backward + optimize

        # fit temperature scalar to validation set
        if include_temperature_scaling:
            model.set_temperature()

        optimizer.zero_grad()

        outputs = model(images.to(device))

        loss = loss_fn(outputs, labels.to(device))
        

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # print statistics
        print('[EPOCH %d / %i, BATCH %5d / %i] regular entropy: %.3f' %
              (epoch + 1, epochs, i + 1, batch_numbers, loss.item()))

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            outputs = model(images.to(device))
            loss = loss_fn(outputs, labels.to(device))

            test_losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(0)

    mean_train_losses.append(np.mean(train_losses))
    # mean_valid_losses.append(np.mean(valid_losses))
    mean_test_losses.append(np.mean(test_losses))

    accuracy = 100 * correct / total
    test_acc_list.append(accuracy)
    test_entropy_list.append(np.mean(test_losses))
    print('epoch : {}, train loss : {:.4f}, test loss : {:.4f}, test acc : {:.2f}%' \
          .format(epoch + 1, np.mean(train_losses),  np.mean(test_losses), accuracy))

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


# plt.plot(range(epochs), test_acc_list)

scaled_string = 'scaled' if include_temperature_scaling else 'no_scaling'

# Plot test accuracy
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy', color=color)
ax1.plot(range(epochs), test_acc_list, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Training Cross-Entropy', color=color)  # we already handled the x-label with ax1
ax2.plot(range(epochs), mean_train_losses, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
plt.savefig(os.getcwd() + '/acc_vs_train_entropy_' + scaled_string + '.png', tight_layout=True, dpi=600)
# plt.savefig()

plt.close()
#

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy', color=color)
ax1.plot(range(epochs), test_acc_list, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Test Cross-Entropy', color=color)  # we already handled the x-label with ax1
ax2.plot(range(epochs), test_entropy_list, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.savefig(os.getcwd() + '/acc_vs_test_entropy_' + scaled_string + '.png', tight_layout=True, dpi=600)



# THE ACTUAL INTERESTING PLOT

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy', color=color)
ax1.plot(range(epochs), test_acc_list, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color1 = 'tab:blue'
color1 = 'tab:green'
ax2.set_ylabel('Cross-Entropy', color=color)  # we already handled the x-label with ax1
ax2.plot(range(epochs), test_entropy_list, color=color1)
ax2.plot(range(epochs), mean_train_losses, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.savefig(os.getcwd() + '/acc_vs_train_and_test_entropy_' + scaled_string + '.png', tight_layout=True, dpi=600)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



# ROLLING MEAN
n = 5

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy', color=color)
ax1.plot(range(epochs)[0:(epochs-n+1)], moving_average(test_acc_list, n=n), color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
color1 = 'tab:green'
ax2.set_ylabel('Cross-Entropy', color=color)  # we already handled the x-label with ax1
ax2.plot(range(epochs)[0:(epochs-n+1)], moving_average(test_entropy_list, n=n), color=color1)
ax2.plot(range(epochs)[0:(epochs-n+1)], moving_average(mean_train_losses, n=n), color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.savefig(os.getcwd() + '/acc_vs_train_and_test_entropy_mean_' + scaled_string + '.png', tight_layout=True, dpi=600)





fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy', color=color)
ax1.plot(range(epochs)[0:(epochs-n+1)], moving_average(test_acc_list, n=n), color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Test Cross-Entropy', color=color)  # we already handled the x-label with ax1
ax2.plot(range(epochs)[0:(epochs-n+1)], moving_average(test_entropy_list, n=n), color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.savefig(os.getcwd() + '/acc_vs_test_entropy_mean_' + scaled_string + '.png', tight_layout=True, dpi=600)
