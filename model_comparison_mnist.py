import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision import datasets, transforms
# from torchvision.models import ResNet
from models import MLP, ModelWithTemperature, MnistResNet, LogisticRegression


from torch.autograd import Variable
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
model_name = 'MLP'


include_temperature_scaling = True
BATCH_SIZE = 256
print_every_n = 200
val_n = 0.1

lr = 0.0001
epochs = 50

SEEDS = range(1, 11)
# SEEDS = range(2)
training_sizes = [25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 40000, 60000]
# training_sizes = [25, 100]



#################




kwargs = {'batch_size': BATCH_SIZE}

# TODO: Add this as a loop
# training_size = training_sizes[0]

for seed in SEEDS:

    results_df = pd.DataFrame(index=range(epochs), columns=training_sizes)

    torch.manual_seed(seed)

    # tempscaler = TempScaler()

    device = torch.device("cuda")

    " DATA SETTINGS "

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transform)

    valid_cross_entropy_full = []
    test_cross_entropy_full = []

    for training_size in training_sizes:

        train_set, _ = torch.utils.data.random_split(train_dataset, [training_size, len(train_dataset) - training_size])  # split into small datasets

        # TODO: Functionality for only selecting small proportion of training data

        n_training = round(len(train_set) * (1-val_n))
        n_val = round(len(train_set) * val_n)
        n_val = n_val if n_val + n_training == training_size else n_val + training_size - (n_val + n_training)

        train_set, val_set = torch.utils.data.random_split(train_set, [n_training, n_val])  #90/10% split

        test_dataset = datasets.MNIST('../data', train=False,
                                      transform=transform)
        batch_numbers = len(train_set) // BATCH_SIZE + 1

        train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
        valid_loader = torch.utils.data.DataLoader(val_set, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

        " MODEL SETTINGS "

        if model_name == 'MLP':
            model = MLP(dropout=True).to(device)
        elif model_name == 'ResNet':
            # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).to(device)
            model = MnistResNet().to(device)
        elif model_name == 'logistic':
            model = LogisticRegression().to(device)

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

            # fit temperature scalar to validation set (only every epoch to save time)
            if include_temperature_scaling:
                model.set_temperature()

            for i, (images, labels) in enumerate(train_loader):
                # if model_name == 'ResNet':
                #     images = Variable(images.resize_(BATCH_SIZE, 1, 32, 32))
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
                    # if model_name == 'ResNet':
                    #     images = Variable(images.resize_(BATCH_SIZE, 1, 32, 32))
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
                  .format(epoch + 1, np.mean(train_losses), np.mean(test_losses), accuracy))

            results_df.iloc[epoch][training_size] = np.mean(test_losses)


        test_cross_entropy_full.append(np.mean(test_entropy_list))

        print('Finished Training with training size = %s' % str(training_size))



    results_df.to_csv(os.getcwd() + '/' + model_name + '_' + 'results_seed_' + str(seed) + '.csv')


# scaled_string = 'scaled' if include_temperature_scaling else 'no_scaling'
# #
# # Plot test accuracy
# fig, ax1 = plt.subplots()
#
# color = 'tab:red'
# ax1.set_xlabel('Training Set Size')
# ax1.set_ylabel('Test Cross-entropy', color=color)
# ax1.plot(range(len(training_sizes)), test_cross_entropy_full, color=color)
# ax1.tick_params(axis='y', labelcolor=color)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
# plt.savefig(os.getcwd() + '/test_CE_vs_datapoints.png', tight_layout=True, dpi=600)


# SAVE RESULTS INTO txt or similar