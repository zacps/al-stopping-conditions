import tarfile
import pickle
import glob

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import fetch_openml
from tabulate import tabulate
from bs4 import BeautifulSoup

# Datasets to implement in order of active learning difficulty (from ALDataset)
# German
# Sonar
# Splice
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


def dataset_summary(names, datasets):
    data = [dataset() for dataset in datasets]
    for i in range(len(data)):
        if len(data[i]) == 4:
            data[i] = (np.concatenate((data[i][0], data[i][2])), np.concatenate((data[i][1], data[i][3])))
    class_prop=[np.unique(y, return_counts=True) for X, y in data]
    
    
    print(tabulate([
        [
            names[i],
            X.shape[0], 
            np.unique(y).shape[0], 
            X.shape[-1], 
            f"{class_prop[i][0][np.argmax(class_prop[i][1])]} {class_prop[i][1][np.argmax(class_prop[i][1])]/X.shape[0]:.0%}", 
            f"{class_prop[i][0][np.argmin(class_prop[i][1])]} {class_prop[i][1][np.argmin(class_prop[i][1])]/X.shape[0]:.0%}"
        ] for i, (X, y) in enumerate(data)
    ], headers=["Dataset", "Instances", "Classes", "Features", "Most common class", "Least common class"]))


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
    y = dataset[60].to_numpy()
    X = dataset.drop([60], axis=1).to_numpy()
    return X, y


def splice():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data"
    dataset = pd.read_csv(url, header=None)
    y = dataset[0].to_numpy()
    X = dataset[2].apply(lambda x: pd.Series(list(x.strip()))).to_numpy()
    X = sklearn.preprocessing.OneHotEncoder().fit_transform(X)
    return X, y

def mfeat():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-pix"
    dataset = pd.read_csv(url, header=None)
    
    # TODO:
    
def bbbp():
    """
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\BloodBrainBarrierPenetration\dataset_cddd.csv", header=0)
    X = dataset.iloc[:,2:514].to_numpy()
    y = dataset["penetration"].to_numpy()
    
    X, _, y, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    
    return X, y

def hiv():
    """
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\HIV\dataset_cddd.csv")
    y = dataset["activity"].to_numpy()
    X = dataset.iloc[:,3:515].to_numpy()
    
    X, _, y, _ = train_test_split(X, y, test_size=0.975, random_state=42)
    
    return X, y

def mutagen():
    """
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\Mutagenicity\dataset_cddd.csv")
    y = dataset['mutagen'].to_numpy()
    X = dataset.iloc[:,2:514].to_numpy()
    
    X, _, y, _ = train_test_split(X, y, test_size=0.84, random_state=42)
    
    return X, y
    
def MUV(dataset_id):
    """
    Contains datasets 466, 548, 600, 644, 652, 689, 692, 712, 713, 733, 737, 810, 832, 846, 852, 858, 859
    
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    assert(dataset_id in [466, 548, 600, 644, 652, 689, 692, 712, 713, 733, 737, 810, 832, 846, 852, 858, 859])
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\MUV\dataset_cddd.csv")
    dataset = dataset[dataset[str(dataset_id)] != '?']
    y = dataset[str(dataset_id)].to_numpy('int')
    X = dataset.iloc[:,dataset.columns.get_loc('cddd_1'):dataset.columns.get_loc('cddd_512')].to_numpy('float')
    
    X, _, y, _ = train_test_split(X, y, train_size=4000, random_state=42)
    
    return X, y
    
def sider():
    """
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\SIDER\dataset_cddd.csv")
    
    y = dataset.iloc[:,1:28].to_numpy()
    encoder = sklearn.preprocessing.LabelBinarizer()
    encoder.fit(list(range(1, 11)))
    y = encoder.inverse_transform(y)
    
    X = dataset.iloc[:,28:546].to_numpy()
    
    X, _, y, _ = train_test_split(X, y, train_size=1000, random_state=42)
    
    return X, y

def tox21(dataset_id):
    """
    Contains datasets:
    ['nr-ahr','nr-ar-lbd','nr-aromatase','nr-ar','nr-er-lbd','nr-er','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
    
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    assert(dataset_id in ['nr-ahr','nr-ar-lbd','nr-aromatase','nr-ar','nr-er-lbd','nr-er','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53'])
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\tox21\dataset_cddd.csv")
    dataset = dataset[dataset[str(dataset_id)] != '?']
    y = dataset[str(dataset_id)].to_numpy('int')
    X = dataset.iloc[:,dataset.columns.get_loc('cddd_1'):dataset.columns.get_loc('cddd_512')].to_numpy('float')
    
    X, _, y, _ = train_test_split(X, y, train_size=1000, random_state=42)
    
    return X, y

def mnist(dataset_size=1000):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    if dataset_size != 70000:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y

def quickdraw():
    """
    Using the same classes as 'Adversarial Active Learning'.
    """
    
    Xs = []
    ys = []
    for i, klass in enumerate(["cat", "dolphin", "angel", "face"]):
        X = np.load(f"datasets/full_numpy_bitmap_{klass}.npy")
        y = np.zeros((X.shape[0], 4))
        y[:,i] = np.ones(X.shape[0])
        Xs.append(X)
        ys.append(y)
        
    X = np.concatenate(Xs)
    X = X.astype('float64')
    y = np.concatenate(ys)
    
    encoder = sklearn.preprocessing.LabelBinarizer()
    encoder.fit(list(range(1, 5)))
    y = encoder.inverse_transform(y)
    
    X, _, y, _ = train_test_split(X, y, train_size=4000, random_state=42)
    
    return X, y
    
    
def newsgroups(categories=None):
    bunch = datasets.fetch_20newsgroups_vectorized(subset='all', remove=('headers'), normalize=True)
    X = bunch.data
    y = bunch.target
    
    if categories:
        cat_map = np.array(bunch.target_names)
        categories_idx = [np.where(cat_map == category) for category in categories]
        idx = np.isin(y, categories_idx)
        y = y[idx]
        X = X[idx]
        

    return X, y


def reuters21578():
    """
    This seems like a nightmare to preprocess...
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz'
    dataset = pd.read_csv(url, compression='gzip', header=None)
    return dataset


def rcv1(category='CCAT'):
    bunch = datasets.fetch_rcv1()
    y = bunch.target.getcol(np.where(bunch.target_names == category)[0][0])
    y = np.squeeze(np.array(y.todense()))
    X = bunch.data
    return X, y
    
    
def cifar10():
    """
    Image recognition dataset.
    
    TODO: The separate train/test return needs to be supported in librun or possibly in active_split.
    
    http://www.cs.toronto.edu/~kriz/cifar.html
    """
    with tarfile.open("datasets/cifar-10-python.tar.gz", "r:gz") as archive:
        filenames = [*[f"cifar-10-batches-py/data_batch_{i}" for i in range(1,6)], 'cifar-10-batches-py/test_batch']
        files = [archive.extractfile(filename) for filename in filenames]
        dicts = [pickle.load(file, encoding='bytes') for file in files]
        
    X_train = np.concatenate([d[b'data'] for d in dicts[:-1]])
    y_train = np.concatenate([d[b'labels'] for d in dicts[:-1]])
    X_test = dicts[-1][b'data']
    y_test = np.array(dicts[-1][b'labels'])
    
    return X_train, y_train, X_test, y_test
        
        
def higgs():
    dataset = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz', compression='gz')
    y = dataset[0].to_numpy()
    X = dataset.iloc[:,1:].to_numpy()
    
    X, _, y, _ = train_test_split(X, y, test_size=0.84, random_state=42)
    
    return X, y


def webkb():
    """
    http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/
    """
    classes = ['course', 'faculty', 'project', 'student']
    universities = ['cornell', 'texas', 'washington', 'wisconsin', 'misc']
    
    docs = []
    for klass_idx, klass in enumerate(classes):
        for uni in universities:
            paths = glob.glob(f'datasets/webkb/{klass}/{uni}/*')
            for path in paths:
                with open(path, 'r') as file:
                    soup = BeautifulSoup(file, features='html5lib')
                docs.append((soup.get_text(), klass))
                
    X = [doc[0] for doc in docs]
    y = [doc[1] for doc in docs]
    
    X = sklearn.feature_extraction.text.CountVectorizer(min_df=2).fit_transform(X)
    
    return X, np.array(y)
    
 
def spamassassin():
    """
    https://spamassassin.apache.org/old/publiccorpus/
    """
    spam = ['datasets/spamassasin/20030228_spam.tar.bz2', 'datasets/spamassasin/20050311_spam_2.tar.bz2']
    ham = [
        'datasets/spamassasin/20030228_easy_ham_2.tar.bz2', 
        'datasets/spamassasin/20030228_easy_ham.tar.bz2', 
        'datasets/spamassasin/20030228_hard_ham.tar.bz2'
    ]
    docs = []
    for filename in spam:
        with tarfile.open(filename, "r:bz2") as archive:
            for file in archive:
                if not file.isfile():
                    continue
                docs.append((archive.extractfile(file).read().decode('latin-1'), 'spam'))
    for filename in ham:
        with tarfile.open(filename, "r:bz2") as archive:
            for file in archive:
                if not file.isfile():
                    continue
                docs.append((archive.extractfile(file).read().decode('latin-1'), 'ham'))
                
    X = [doc[0] for doc in docs]
    y = [doc[1] for doc in docs]
    X = sklearn.feature_extraction.text.CountVectorizer(min_df=2).fit_transform(X)
    
    return X, np.array(y)