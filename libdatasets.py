import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Datasets to implement in order of active learning difficulty (from ALDataset)
# German (untested)
# Sonar (untested)
# Splice (untested)
# Clean1 (multiple instance problem, complicating factor?)
# Diabetes
# Australian
# Heart
# Ex8b
# Vehicle (split into a million files, but has a big accuracy change over time in ALDataset testing)
# Ex8a
# Gcloudub
# Ionosphere
# XOR
# Gcloudb


def banknote():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    dataset = pd.read_csv(url, header=None)
    y = dataset[4].to_numpy()
    isInB = np.array([dataset.to_numpy()[i, 0] > 0.32 for i in range(len(dataset))])
    isInB = isInB.reshape(len(isInB), 1)
    X = dataset.drop([4], axis=1).to_numpy()
    # pca = PCA(n_components=21).fit(X)
    # X = pca.transform(X)
    X = np.append(X, isInB, axis=1)

    return X, y


def bias_banknote(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


def haberman():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    dataset = pd.read_csv(url, header=None)

    y = dataset[4].to_numpy()
    X = dataset.drop([4], axis=1).to_numpy()

    return X, y


def digits():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target


def abalone():
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    )
    dataset = pd.read_csv(url, header=None)
    y = dataset[0].to_numpy()
    isInB = np.array([dataset.to_numpy()[i, 6] < 0.144 for i in range(len(dataset))])
    isInB = isInB.reshape(len(isInB), 1)
    X = dataset.drop([0, 6], axis=1).to_numpy()
    X = np.append(X, isInB, axis=1)

    return X, y


def bias_abalone(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


def car():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    toNum = {
        "low": 1,
        "med": 2,
        "high": 3,
        "vhigh": 4,
        "5more": 5,
        "more": 5,
        "small": 1,
        "big": 3,
    }
    dataset = pd.read_csv(url, header=None)
    y = dataset[6].to_numpy()
    dataset = dataset.drop([6], axis=1)
    dataset = dataset.replace(
        {0: toNum, 1: toNum, 2: toNum, 3: toNum, 4: toNum, 5: toNum}
    )
    dataset = dataset.apply(pd.to_numeric)

    isInB = np.array([dataset.to_numpy()[i, 3] > 3 for i in range(len(dataset))])
    isInB = isInB.reshape(len(isInB), 1)
    # dataset = dataset.drop([3], axis=1)
    X = np.append(dataset.to_numpy(), isInB, axis=1)

    return X, y


def bias_car(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


def cardio():
    dataset = pd.read_csv(
        "Imitate/Datasets/cardio_train.csv", header=0, sep=";", index_col=0
    )
    y = dataset["cardio"].to_numpy()
    dataset = dataset[["age", "weight"]]
    dataset = dataset.assign(age=dataset.age / 365.25)

    isInB = np.array([1] * len(dataset))
    isInB = isInB.reshape(len(isInB), 1)

    X = np.append(dataset.to_numpy(), isInB, axis=1)

    # draw sample
    X, _, y, _ = train_test_split(X, y, test_size=0.985)

    return X, y


def bias_cardio(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


def shuttle(dataset_size=58000):
    dataset1 = pd.read_csv("Imitate/Datasets/shuttle.trn", header=None, sep="\s")
    dataset2 = pd.read_csv("Imitate/Datasets/shuttle.tst", header=None, sep="\s")
    dataset = np.concatenate((dataset1, dataset2))
    y = dataset[:, -1] == 1
    isInB = np.array([dataset[i, 0] > 54.5 for i in range(len(dataset))])
    isInB = isInB.reshape(len(isInB), 1)
    X = dataset[:, 0:-1]
    # pca = PCA(n_components=21).fit(X)
    # X = pca.transform(X)
    X = np.append(X, isInB, axis=1)

    # draw sample
    X, _, y, _ = train_test_split(X, y, test_size=0.8)

    return X, y


def bias_shuttle(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


def skin(dataset_size=4177):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt"
    dataset = pd.read_csv(url, header=None, sep="\t")
    y = dataset[3].to_numpy()
    isInB = np.array([dataset.to_numpy()[i, 2] <= 170.5 for i in range(len(dataset))])
    isInB = isInB.reshape(len(isInB), 1)
    X = dataset.drop([2, 3], axis=1).to_numpy()
    # pca = PCA(n_components=21).fit(X)
    # X = pca.transform(X)
    X = np.append(X, isInB, axis=1)

    # draw sample
    X, _, y, _ = train_test_split(X, y, test_size=0.95, random_state=42)

    return X, y


def bias_skin(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


def german():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
    dataset = pd.read_csv(url, header=None, delim_whitespace=True)
    y = dataset[24].to_numpy()
    X = dataset.drop([24], axis=1).to_numpy()
    return X, y


def sonar():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    dataset = pd.read_csv(url, header=None)
    y = dataset[-1].to_numpy()
    X = dataset.drop([-1], axis=1).to_numpy()
    return X, y


def splice():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data"
    dataset = pd.read_csv(url, header=None)
    y = dataset[-1].to_numpy()
    X = dataset.drop([-1], axis=1).to_numpy()
    return X, y
