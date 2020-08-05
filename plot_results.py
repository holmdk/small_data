import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd() + '/results/'


" REGULAR VERSION "
files = glob.glob(path + '*.csv')

models = ['MLP_no_dropout', 'logistic', 'ResNet']

for model in models:

    load_files = glob.glob(path + model + '_results_seed_*.csv')
    data = pd.read_csv(load_files[0], index_col=0)
    data[:] = 0

    for seed in load_files:
        temp = pd.read_csv(seed, index_col=0)
        data += temp

    data /= len(load_files)

    means = data.mean()
    stdevs = data.std()



    plt.plot(means, label=model)
    plt.fill_between(range(15), means - stdevs, means + stdevs, alpha=0.1, color='gray')


plt.legend(loc="upper left")
plt.show()





" IMBALANCED VERSION "
files = glob.glob(path + '*.csv')

models = ['imbalanced_no_dropout_MLP', 'imbalanced_MLP', 'imbalanced_logistic', 'imbalanced_ResNet']

for model in models:

    load_files = glob.glob(path + model + '_results_seed_*.csv')
    data = pd.read_csv(load_files[0], index_col=0)
    data[:] = 0

    for seed in load_files:
        temp = pd.read_csv(seed, index_col=0)
        data += temp

    data /= len(load_files)

    means = data.mean()
    stdevs = data.std()



    plt.plot(means, label=model)
    plt.fill_between(range(15), means - stdevs, means + stdevs, alpha=0.1, color='gray')


plt.legend(loc="upper left")
plt.show()