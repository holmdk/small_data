# make imbalanced data
# from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/examples/mnist.ipynb
import torch
from torchvision import datasets, transforms
import copy
import random
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision import datasets, transforms
# from torchvision.models import ResNet
import matplotlib.pyplot as plt
import seaborn as sns
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

def sample_imbalance(data_loader, dataset):
    idx_to_del = [i for i, label in enumerate(data_loader.dataset.targets)
                  if random.random() > sample_probs[label]]

    imbalanced_dataset = copy.deepcopy(dataset)
    imbalanced_dataset.targets = np.delete(data_loader.dataset.targets, idx_to_del, axis=0)
    imbalanced_dataset.data = np.delete(data_loader.dataset.data, idx_to_del, axis=0)

    return imbalanced_dataset

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

kwargs = {'batch_size': BATCH_SIZE}




transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # transforms.Normalize((0.1307,), (0.3081,))
])



num_classes = 10
classe_labels = range(num_classes)
sample_probs = torch.rand(num_classes)


for seed in SEEDS:

    results_df = pd.DataFrame(index=range(epochs), columns=training_sizes)

    torch.manual_seed(seed)

    # tempscaler = TempScaler()

    device = torch.device("cuda")

    valid_cross_entropy_full = []
    test_cross_entropy_full = []

    for training_size in training_sizes:
        print(training_size)
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                                       transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, **kwargs)

        imbalanced_train_dataset = sample_imbalance(train_loader, train_dataset)
        imbalanced_train_dataset, _ = torch.utils.data.random_split(imbalanced_train_dataset, [training_size, len(
            imbalanced_train_dataset) - training_size])  # split into small datasets

        # TODO: Functionality for only selecting small proportion of training data

        n_training = round(len(imbalanced_train_dataset) * (1 - val_n))
        n_val = round(len(imbalanced_train_dataset) * val_n)
        n_val = n_val if n_val + n_training == len(imbalanced_train_dataset) else n_val + n_training - (
                    n_val + n_training)

        n_val = n_val if n_val + n_training == len(imbalanced_train_dataset) else n_val + (len(imbalanced_train_dataset) - (n_val + n_training))

        imbalanced_train_dataset, imbalanced_val_dataset = torch.utils.data.random_split(imbalanced_train_dataset,
                                                                                         [n_training,
                                                                                          n_val])  # 90/10% split

        train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, **kwargs)
        valid_loader = torch.utils.data.DataLoader(imbalanced_val_dataset, **kwargs)

        # TEST DATA
        test_dataset = datasets.MNIST('../data', train=False,
                                      transform=transform)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **kwargs)

        test_dataset = sample_imbalance(test_dataloader, test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

        batch_numbers = len(imbalanced_train_dataset) // BATCH_SIZE + 1


        if model_name == 'MLP':
            model = MLP(dropout=False).to(device)
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

    if model_name == 'MLP':
        results_df.to_csv(os.getcwd() + '/imbalanced_no_dropout_' + model_name + '_' + 'results_seed_' + str(seed) + '.csv')
    else:
        results_df.to_csv(os.getcwd() + '/imbalanced_' + model_name + '_' + 'results_seed_' + str(seed) + '.csv')































# VISUALIZATION
def show_mnist(arr, nrow=5, ncol=10, figsize=None):
    if figsize is None:
        figsize = (ncol, nrow)

    f, a = plt.subplots(nrow, ncol, figsize=figsize)

    def _do_show(the_figure, the_array):
        the_figure.imshow(the_array)
        the_figure.axis('off')

    for i in range(nrow):
        for j in range(ncol):
            _do_show(a[i][j], np.reshape(arr[i * ncol + j], (28, 28)))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.draw()
    plt.savefig(os.getcwd() + '/mnist_show.png', tight_layout=True, dpi=600)



# def vis(test_accs, confusion_mtxes, labels, figsize=(20, 8)):
#     cm = confusion_mtxes[np.argmax(test_accs)]
#     cm_sum = np.sum(cm, axis=1, keepdims=True)
#     cm_perc = cm / cm_sum * 100
#     annot = np.empty_like(cm).astype(str)
#     nrows, ncols = cm.shape
#     for i in range(nrows):
#         for j in range(ncols):
#             c = cm[i, j]
#             p = cm_perc[i, j]
#             if c == 0:
#                 annot[i, j] = ''
#             else:
#                 annot[i, j] = '%.1f%%' % p
#     cm = pd.DataFrame(cm, index=labels, columns=labels)
#     cm.index.name = 'Actual'
#     cm.columns.name = 'Predicted'
#
#     fig = plt.figure(figsize=figsize)
#     plt.subplot(1, 2, 1)
#     plt.plot(test_accs, 'g')
#     plt.grid(True)
#
#     plt.subplot(1, 2, 2)
#     sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
#
#     plt.savefig(os.getcwd() + '/acc_vs_test_entropy_' + scaled_string + '.png', tight_layout=True, dpi=600)
#
#     plt.show()

#
# print('Original dataset: %d training samples & %d testing samples\n' % (
#     len(train_loader.dataset), len(test_loader.dataset)))
#
# print('Distribution of classes in original dataset:')
# fig, ax = plt.subplots()
# _, counts = np.unique(train_dataset.targets, return_counts=True)
# ax.bar(classe_labels, counts)
# ax.set_xticks(classe_labels)
#
# plt.savefig(os.getcwd() + '/distribution_original.png', tight_layout=True, dpi=600)
#
#
# print('Sampling probability for each class:')
# fig, ax = plt.subplots()
# ax.bar(classe_labels, sample_probs)
# ax.set_xticks(classe_labels)
# plt.savefig(os.getcwd() + '/sampling_probability.png', tight_layout=True, dpi=600)
# plt.show()
#
#
# print('Imbalanced dataset: %d training samples & %d testing samples\n' % (
#     len(train_loader.dataset), len(test_loader.dataset)))
#
# print('Distribution of classes in imbalanced dataset:')
# fig, ax = plt.subplots()
# _, counts = np.unique(imbalanced_train_dataset.dataset.dataset.targets, return_counts=True)
# ax.bar(classe_labels, counts)
# ax.set_xticks(classe_labels)
# # plt.show()
# plt.savefig(os.getcwd() + '/distribution_imbalanced.png', tight_layout=True, dpi=600)
#
# for data, _ in train_loader:
#     show_mnist(data)
#     break