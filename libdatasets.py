import tarfile
import pickle
import glob
from functools import partial, lru_cache
import requests, zipfile, io
from os.path import exists

import scipy
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from bs4 import BeautifulSoup

from libutil import dataset_dir

def dataset_summary(names, datasets):
    data = [dataset() for dataset in datasets]
    class_prop=[np.unique(y, return_counts=True) for _, y in data]

    print(tabulate([
        [
            names[i],
            X.shape[0], 
            np.unique(y).shape[0], 
            X.shape[-1], 
            f"{class_prop[i][0][np.argmax(class_prop[i][1])]} {class_prop[i][1][np.argmax(class_prop[i][1])]/X.shape[0]:.0%}", 
            f"{class_prop[i][0][np.argmin(class_prop[i][1])]} {class_prop[i][1][np.argmin(class_prop[i][1])]/X.shape[0]:.0%}",
            getattr(datasets[i], "domain", "general")
        ] for i, (X, y) in enumerate(data)
    ], headers=["Dataset", "Instances", "Classes", "Features", "Most common class", "Least common class", "Domain"]))

    
def attr(name, value):
    def decorator(func):
        setattr(func, name, value)
        return func
    return decorator
    
domain = partial(attr, "domain")
source = partial(attr, "source")


# ------------------------------------------------------------------------------------------------------------------------------

@source("https://archive.ics.uci.edu/ml/datasets/banknote+authentication")
def banknote(dataset_size=1000):
    cache = _cache_restore("banknote")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    dataset = pd.read_csv(url, header=None)
    y = dataset[4].to_numpy()
    isInB = np.array([dataset.to_numpy()[i, 0] > 0.32 for i in range(len(dataset))])
    isInB = isInB.reshape(len(isInB), 1)
    X = dataset.drop([4], axis=1).to_numpy()
    # pca = PCA(n_components=21).fit(X)
    # X = pca.transform(X)
    X = np.append(X, isInB, axis=1)


    _cache_save("banknote", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


def bias_banknote(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/haberman's+survival")
def haberman(dataset_size=1000):
    cache = _cache_restore("haberman")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    dataset = pd.read_csv(url, header=None)

    y = dataset[4].to_numpy()
    X = dataset.drop([4], axis=1).to_numpy()

    _cache_save("haberman", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


def digits():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target


@source("https://archive.ics.uci.edu/ml/datasets/abalone")
def abalone(dataset_size=1000):
    cache = _cache_restore("abalone")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    )
    dataset = pd.read_csv(url, header=None)
    y = dataset[0].to_numpy()
    isInB = np.array([dataset.to_numpy()[i, 6] < 0.144 for i in range(len(dataset))])
    isInB = isInB.reshape(len(isInB), 1)
    X = dataset.drop([0, 6], axis=1).to_numpy()
    X = np.append(X, isInB, axis=1)


    _cache_save("abalone", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


def bias_abalone(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/car+evaluation")
def car(dataset_size=1000):
    cache = _cache_restore("car")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
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

    _cache_save("car", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


def bias_car(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


def cardio(dataset_size=1000):
    cache = _cache_restore("cardio")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        X, y = _normalize(X, y)
        return X, y
    
    dataset = pd.read_csv(
        "Imitate/Datasets/cardio_train.csv", header=0, sep=";", index_col=0
    )
    y = dataset["cardio"].to_numpy()
    dataset = dataset[["age", "weight"]]
    dataset = dataset.assign(age=dataset.age / 365.25)

    isInB = np.array([1] * len(dataset))
    isInB = isInB.reshape(len(isInB), 1)

    X = np.append(dataset.to_numpy(), isInB, axis=1)

    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)

    _cache_save("cardio", X, y)
    X, y = _split(X, y, dataset_size)
    X, y = _normalize(X, y)
    return X, y


def bias_cardio(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


def shuttle(dataset_size=1000):
    cache = _cache_restore("shuttle")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        X, y = _normalize(X, y)
        return X, y
    
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

    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)

    _cache_save("shuttle", X, y)
    X, y = _split(X, y, dataset_size)
    X, y = _normalize(X, y)
    return X, y


def bias_shuttle(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/skin+segmentation")
def skin(dataset_size=1000):
    cache = _cache_restore("skin")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        X, y = _normalize(X, y)
        return X, y
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt"
    dataset = pd.read_csv(url, header=None, sep="\t")
    y = dataset[3].to_numpy()
    isInB = np.array([dataset.to_numpy()[i, 2] <= 170.5 for i in range(len(dataset))])
    isInB = isInB.reshape(len(isInB), 1)
    X = dataset.drop([2, 3], axis=1).to_numpy()
    # pca = PCA(n_components=21).fit(X)
    # X = pca.transform(X)
    X = np.append(X, isInB, axis=1)

    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)

    _cache_save("skin", X, y)
    X, y = _split(X, y, dataset_size)
    X, y = _normalize(X, y)
    return X, y


def bias_skin(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)")
def german(dataset_size=1000):
    cache = _cache_restore("german")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size if dataset_size < 1000 else None)
        return X, y
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
    dataset = pd.read_csv(url, header=None, delim_whitespace=True)
    y = dataset[24].to_numpy()
    X = dataset.drop([24], axis=1).to_numpy()
    _cache_save("german", X, y)
    X, y = _split(X, y, dataset_size if dataset_size < 1000 else None)
    return X, y


@source("http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)")
def sonar(dataset_size=1000):
    cache = _cache_restore("sonar")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    dataset = pd.read_csv(url, header=None)
    y = dataset[60].to_numpy()
    X = dataset.drop([60], axis=1).to_numpy()
    _cache_save("sonar", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)")
def splice(dataset_size=1000):
    cache = _cache_restore("splice")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data"
    dataset = pd.read_csv(url, header=None)
    y = dataset[0].to_numpy()
    X = dataset[2].apply(lambda x: pd.Series(list(x.strip()))).to_numpy()
    X = sklearn.preprocessing.OneHotEncoder().fit_transform(X)

    
    _cache_save("splice", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


def mfeat():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-pix"
    dataset = pd.read_csv(url, header=None)
    
    # TODO:
    
    
@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
def bbbp(dataset_size=1000):
    """
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    cache = _cache_restore("bbbp")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    dataset = pd.read_csv(r"datasets/bbbp/dataset_cddd.csv", header=0)
    X = dataset.iloc[:,2:514].to_numpy()
    y = dataset["penetration"].to_numpy()

    
    _cache_save("bbbp", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
def hiv(dataset_size=1000):
    """
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    cache = _cache_restore("hiv")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\HIV\dataset_cddd.csv")
    y = dataset["activity"].to_numpy()
    X = dataset.iloc[:,3:515].to_numpy()

    
    _cache_save("hiv", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
def mutagen(dataset_size=1000):
    """
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    cache = _cache_restore("mutagen")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\Mutagenicity\dataset_cddd.csv")
    y = dataset['mutagen'].to_numpy()
    X = dataset.iloc[:,2:514].to_numpy()

    
    _cache_save("mutagen", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
def MUV(dataset_id, dataset_size=1000):
    """
    Contains datasets 466, 548, 600, 644, 652, 689, 692, 712, 713, 733, 737, 810, 832, 846, 852, 858, 859
    
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    cache = _cache_restore("MUV")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    assert(dataset_id in [466, 548, 600, 644, 652, 689, 692, 712, 713, 733, 737, 810, 832, 846, 852, 858, 859])
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\MUV\dataset_cddd.csv")
    dataset = dataset[dataset[str(dataset_id)] != '?']
    y = dataset[str(dataset_id)].to_numpy('int')
    X = dataset.iloc[:,dataset.columns.get_loc('cddd_1'):dataset.columns.get_loc('cddd_512')].to_numpy('float')

    
    _cache_save("MUV", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y
  
    
@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
def sider(dataset_size=1000):
    """
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    cache = _cache_restore("sider")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\SIDER\dataset_cddd.csv")
    
    y = dataset.iloc[:,1:28].to_numpy()
    encoder = sklearn.preprocessing.LabelBinarizer()
    encoder.fit(list(range(1, 11)))
    y = encoder.inverse_transform(y)
    
    X = dataset.iloc[:,28:546].to_numpy()
    
    X, _, y, _ = train_test_split(X, y, train_size=1000, random_state=42)
    
    _cache_save("sider", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
def tox21(dataset_id, dataset_size=1000):
    """
    Contains datasets:
    ['nr-ahr','nr-ar-lbd','nr-aromatase','nr-ar','nr-er-lbd','nr-er','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
    
    Performance saturates immediately on a linear svm, not a good active learning target.
    
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """

    cache = _cache_restore("tox21")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    assert(dataset_id in ['nr-ahr','nr-ar-lbd','nr-aromatase','nr-ar','nr-er-lbd','nr-er','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53'])
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\tox21\dataset_cddd.csv")
    dataset = dataset[dataset[str(dataset_id)] != '?']
    y = dataset[str(dataset_id)].to_numpy('int')
    X = dataset.iloc[:,dataset.columns.get_loc('cddd_1'):dataset.columns.get_loc('cddd_512')].to_numpy('float')
    
    X, _, y, _ = train_test_split(X, y, train_size=1000, random_state=42)
    
    _cache_save("tox21", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@domain("image")
def mnist(dataset_size=1000, digits=None):
    cache = _cache_restore("mnist")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    if digits is not None:
        assert len(digits) == 2
        idx = np.where((y == str(digits[0])) | (y == str(digits[1])))
        X = X[idx]
        y = y[idx]

    
    _cache_save("mnist", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@domain("image")
@source("https://github.com/googlecreativelab/quickdraw-dataset")
def quickdraw(dataset_size=1000, classes=("cat", "dolphin", "angel", "face")):
    """
    Using the same classes as 'Adversarial Active Learning'.
    """

    cache = _cache_restore("quickdraw")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    
    Xs = []
    ys = []
    for i, klass in enumerate(classes):
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

    
    _cache_save("quickdraw", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y
    
    
@source("https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html")
@domain("nlp")
def newsgroups(dataset_size=1000, categories=None):
    bunch = datasets.fetch_20newsgroups_vectorized(subset='all', remove=('headers'), normalize=True)
    X = bunch.data
    y = bunch.target
    
    if categories:
        cat_map = np.array(bunch.target_names)
        categories_idx = [np.where(cat_map == category) for category in categories]
        idx = np.isin(y, categories_idx)
        y = y[idx]
        X = X[idx]
    
    X, y = _split(X, y, dataset_size)
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection")
@domain("nlp")
def reuters21578(dataset_size=1000):
    """
    """
    cache = _cache_restore("reuters21578")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz'
    dataset = pd.read_csv(url, compression='gzip', header=None)

    
    _cache_save("reuters21578", X, y)
    X, y = _split(X, y, dataset_size)
    return dataset


@domain("nlp")
def rcv1(dataset_size=1000, category='CCAT'):
    cache = _cache_restore("rcv1")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    bunch = datasets.fetch_rcv1()
    y = bunch.target.getcol(np.where(bunch.target_names == category)[0][0])
    y = np.squeeze(np.array(y.todense()))
    X = bunch.data

    
    _cache_save("rcv1", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y
    
    
@source("http://www.cs.toronto.edu/~kriz/cifar.html")
@domain("image")
def cifar10(dataset_size=1000):
    """
    Image recognition dataset.
    
    TODO: The separate train/test return needs to be supported in librun or possibly in active_split.
    
    http://www.cs.toronto.edu/~kriz/cifar.html
    """

    cache = _cache_restore("cifar10")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    with tarfile.open("datasets/cifar-10-python.tar.gz", "r:gz") as archive:
        filenames = [*[f"cifar-10-batches-py/data_batch_{i}" for i in range(1,6)], 'cifar-10-batches-py/test_batch']
        files = [archive.extractfile(filename) for filename in filenames]
        dicts = [pickle.load(file, encoding='bytes') for file in files]
        
    X_train = np.concatenate([d[b'data'] for d in dicts[:-1]])
    y_train = np.concatenate([d[b'labels'] for d in dicts[:-1]])
    X_test = dicts[-1][b'data']
    y_test = np.array(dicts[-1][b'labels'])
    
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    
    _cache_save("cifar10", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y
        
      
@domain("physics")
def higgs(dataset_size=1000):
    cache = _cache_restore("higgs")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    dataset = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz', compression='gz')
    y = dataset[0].to_numpy()
    X = dataset.iloc[:,1:].to_numpy()
    
    X, _, y, _ = train_test_split(X, y, test_size=0.84, random_state=42)
    
    _cache_save("higgs", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source("http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/")
@domain("nlp")
def webkb(dataset_size=1000):
    """
    http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/
    """

    cache = _cache_restore("webkb")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
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
    y = np.array([doc[1] for doc in docs])
    
    X = sklearn.feature_extraction.text.CountVectorizer(min_df=2).fit_transform(X)

    
    _cache_save("webkb", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source("https://spamassassin.apache.org/old/publiccorpus/")
@domain("nlp")
def spamassassin(dataset_size=1000):
    """
    https://spamassassin.apache.org/old/publiccorpus/
    """

    cache = _cache_restore("spamassassin")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
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
    y = np.array([doc[1] for doc in docs])
    X = sklearn.feature_extraction.text.CountVectorizer(min_df=2).fit_transform(X)

    
    _cache_save("spamassassin", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source("http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions")
def smartphone(dataset_size=1000):
    """
    http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
    """
    
    cache = _cache_restore("smartphone")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        y = np.squeeze(y)
        return X, y
    # Set header=None?
    X_train = pd.read_csv('datasets/smartphone/Train/X_train.txt', sep=' ')
    X_test = pd.read_csv('datasets/smartphone/Test/X_test.txt', sep=' ')
    y_train = pd.read_csv('datasets/smartphone/Train/y_train.txt')
    y_test = pd.read_csv('datasets/smartphone/Test/y_test.txt')
    
    X = np.concatenate((X_train.to_numpy(), X_test.to_numpy()), axis=0)
    y = np.concatenate((y_train.to_numpy(), y_test.to_numpy()), axis=0)


    _cache_save("smartphone", X, y)
    X, y = _split(X, y, dataset_size)
    y = np.squeeze(y)
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Covertype")
def covertype(dataset_size=1000):
    """
    https://archive.ics.uci.edu/ml/datasets/Covertype
    """
    
    cache = _cache_restore("covertype")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        X, y = _normalize(X, y)
        return X, y
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz', header=None)
    y = data.iloc[:,-1].to_numpy()
    X = data.iloc[:,0:-1].to_numpy()

    
    _cache_save("covertype", X, y)
    X, y = _split(X, y, dataset_size)
    X, y = _normalize(X, y)
    return X, y
    
    
@domain("physical")
def htru2(dataset_size=1000):
    cache = _cache_restore("htru2")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        X, y = _normalize(X, y)
        return X, y
    data = pd.read_csv('datasets/HTRU2/HTRU_2.csv', header=None)
    y = data.iloc[:,-1].to_numpy()
    X = data.iloc[:,0:-1].to_numpy()

        
    _cache_save("htru2", X, y)
    X, y = _split(X, y, dataset_size)
    X, y = _normalize(X, y)
    return X, y


# Doesn't contain enough rows if excluding missing values.
#@domain("computer")
#def ida2016(dataset_size=1000):
#    test = pd.read_csv('datasets/IDA2016Challenge/aps_failure_test_set.csv', header=14, sep=',')
#    train = pd.read_csv('datasets/IDA2016Challenge/aps_failure_training_set.csv', header=14, sep=',')
#    
#    y = np.concatenate((test.iloc[:,0].to_numpy(), train.iloc[:,0].to_numpy()))
#    X = np.concatenate((test.iloc[:,1:].to_numpy(), train.iloc[:,1:].to_numpy()))
#    
#    # filter missing values
#    idx = np.all(X!='na',axis=1)
#    X = X[idx]
#    y = y[idx]
#    
#    if dataset_size is not None:
#        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
#        
#    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Detect+Malware+Types")
@domain("computer")
def malware(dataset_size=1000):
    cache = _cache_restore("malware")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        X, y = _normalize(X, y)
        return X, y
    cols = set(pd.read_csv('datasets/malware/staDynVt2955Lab.csv').columns) & set(pd.read_csv('datasets/malware/staDynVxHeaven2698Lab.csv').columns) & set(pd.read_csv('datasets/malware/staDynBenignLab.csv').columns)
    
    files = ['datasets/malware/staDynVt2955Lab.csv', 'datasets/malware/staDynBenignLab.csv', 'datasets/malware/staDynVxHeaven2698Lab.csv']
    
    data = pd.concat([pd.read_csv(file)[cols] for file in files])
    
    y = data.label
    X = data.drop('label', axis=1)
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X.to_numpy(), y.to_numpy(), train_size=dataset_size, random_state=42)
        
    _cache_save("malware", X, y)
    X, y = _split(X, y, dataset_size)
    X, y = _normalize(X, y)
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Shill+Bidding+Dataset")
def bidding(dataset_size=1000):
    """
    https://archive.ics.uci.edu/ml/datasets/Shill+Bidding+Dataset
    """
    cache = _cache_restore("bidding")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00562/Shill%20Bidding%20Dataset.csv')
    X = data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X.to_numpy(), y.to_numpy(), train_size=dataset_size, random_state=42)
        
    X = sklearn.preprocessing.OneHotEncoder().fit_transform(X)
        
    _cache_save("bidding", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Swarm+Behaviour")
def swarm(dataset_size=1000, predict='flocking'):
    cache = _cache_restore("swarm")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    data = pd.read_csv(f'datasets/swarm/{predict.capitalize()}.csv')
    data = data.drop(24015)
    data.x1 = data.x1.astype(float)
    X = data.iloc[:,0:-1].to_numpy()
    y = data.iloc[:,-1].to_numpy()

        
    _cache_save("swarm", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y
    
    
@source('https://archive.ics.uci.edu/ml/datasets/Bank+Marketing')
def bank(dataset_size=1000):
    cache = _cache_restore("bank")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip', stream=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        data = pd.read_csv(z.open('bank-full.csv'), sep=';')
    
    X = data.iloc[:,0:-1]
    X = sklearn.preprocessing.OneHotEncoder().fit_transform(X)
    
    y = data.iloc[:,-1]

        
    _cache_save("bank", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source('https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29')
def anuran(dataset_size=1000):
    cache = _cache_restore("anuran")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip', stream=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        data = pd.read_csv(z.open('Frogs_MFCCs.csv'))
    y = data.Species.to_numpy()
    X = data.iloc[:,:-4].to_numpy()

        
    _cache_save("anuran", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Avila")
def avila(dataset_size=1000):
    cache = _cache_restore("avila")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip', stream=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        data1 = pd.read_csv(z.open('avila/avila-tr.txt'), header=None)
        data2 = pd.read_csv(z.open('avila/avila-ts.txt'), header=None)
    data = pd.concat((data1, data2))
    y = data.iloc[:,-1].to_numpy()
    X = data.iloc[:,:-1].to_numpy()

        
    _cache_save("avila", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y


@source('https://archive.ics.uci.edu/ml/datasets/Bach+Choral+Harmony')
def coral(dataset_size=1000):
    cache = _cache_restore("coral")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00298/jsbach_chorals_harmony.zip', stream=True)
    
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        data = pd.read_csv(z.open('jsbach_chorals_harmony.data'), header=None)
    
    y = data.iloc[:,-1]
    X = data.iloc[:,:-1]

        
    _cache_save("coral", X, y)
    X, y = _split(X, y, dataset_size)
    return X, y
    
    
@source('http://ama.liglab.fr/resourcestools/datasets/buzz-prediction-in-social-media/')
def buzz(site, dataset_size=1000):
    cache = _cache_restore("buzz")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        X, y = _normalize(X, y)
        return X, y
    path = 'datasets/buzz/classification.tar.gz'
    if not exists(path):
        r = requests.get('http://ama.liglab.fr/data/buzz/classification.tar.gz', stream=True)
        with open(path, 'wb') as f:
            f.write(r.content)
    with tarfile.open(path, mode="r:") as z:
        data_th = pd.read_csv(z.extractfile('classification/TomsHardware/Absolute_labeling/TomsHardware-Absolute-Sigma-500.data'), header=None)
        data_tw = pd.read_csv(z.extractfile('classification/Twitter/Absolute_labeling/Twitter-Absolute-Sigma-500.data'), header=None)
        
    if site == 'th':
        data = data_th
    elif site == 'tw':
        data = data_tw
    else:
        raise Exception('Invalid site, must be "th" or "tw"')
    
    y = data.iloc[:,-1]
    X = data.iloc[:,:-1]

        
    _cache_save("buzz", X, y)
    X, y = _split(X, y, dataset_size)
    X, y = _normalize(X, y)
    return X, y


@source('https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis')
def sensorless(dataset_size=1000):
    cache = _cache_restore("sensorless")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        X, y = _normalize(X, y)
        return X, y
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt', sep=' ', header=None)
    X = data.iloc[:,:-1].to_numpy()
    y = data.iloc[:,-1].to_numpy()

        
    _cache_save("sensorless", X, y)
    X, y = _split(X, y, dataset_size)
    X, y = _normalize(X, y)
    return X, y


@source('https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results')
def dota2(dataset_size=1000):
    cache = _cache_restore("dota2")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y

    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip', stream=True)
    
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        data1 = pd.read_csv(z.open('dota2Test.csv'), header=None)
        data2 = pd.read_csv(z.open('dota2Train.csv'), header=None)
        
    data = pd.concat((data1,data2))
    
    y = data.iloc[:,0].to_numpy()
    X = data.iloc[:,1:].to_numpy()

    
    _cache_save("dota2", X, y)
    X, y = _split(X, y, dataset_size)
    return X,y


#@source('https://archive.ics.uci.edu/ml/datasets/FMA%3A+A+Dataset+For+Music+Analysis')
#def fma(dataset_size=1000):
#    path = 'datasets/fma/fma_metadata.zip'
#    if not exists(path):
#        r = requests.get('https://os.unil.cloud.switch.ch/fma/fma_metadata.zip', stream=True)
#        with open(path, 'wb') as f:
#            f.write(r.content)
#    with zipfile.ZipFile(path) as z:
#        tracks = pd.read_csv(z.open('fma_metadata/tracks.csv'))
#        features = pd.read_csv(z.open('fma_metadata/features.csv'))
#    return tracks, features


@source('https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset')
def gas(dataset_size=1000):
    cache = _cache_restore("gas")
    if cache is not None:
        X, y = _split(cache[0], cache[1], dataset_size)
        return X, y

    path = 'datasets/gas/Dataset.zip'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip'
    if not exists(path):
        r = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            f.write(r.content)
    with zipfile.ZipFile(path) as z:
        datas = [datasets.load_svmlight_file(z.open(f'Dataset/batch{i}.dat')) for i in range(1,11)]
        X = np.concatenate([data[0].todense() for data in datas])
        y = np.concatenate([data[1] for data in datas])
        
    _cache_save("gas", X, y)
    X, y = _split(X, y, dataset_size)
        
    return X, y
    

# -------------------------------------------------------------------------------------------------


def _split(X, y, size):
    if size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=size, random_state=42)
    return X, y


def _normalize(X, y):
    X = StandardScaler(with_mean=not isinstance(X, scipy.sparse.csr_matrix)).fit_transform(X)
    return X, y

    
def bias_dataset(X_train, X_test, y_train, y_test, rand=None, **kwargs):
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    feature_idx = np.argmax(clf.feature_importances_)
    median = np.median(X_train[:,feature_idx])
    lower = lower = np.nonzero(X_train[:,feature_idx]<median)[0]
    keep = rand.choice(lower, lower.shape[0] // 10)
    idx = np.concatenate((keep, np.nonzero(X_train[:,feature_idx]>=median)[0]))
    return X_train[idx], X_test, y_train[idx], y_test


def _cache_save(name, X, y):
    if type(X) is scipy.sparse.csr_matrix:
        np.savez_compressed(f"{dataset_dir()}/{name}.npz", data=X.data, indices=X.indices, indptr=X.indptr, y=y)
    else:
        np.savez_compressed(f"{dataset_dir()}/{name}.npz", X=X, y=y)

def _cache_restore(name):
    try:
        with np.load(f"{dataset_dir()}/{name}.npz", allow_pickle=True) as f:
            if 'data' in f:
                X = scipy.sparse.csr_matrix((f['data'], f['indices'], f['indptr']))
            else:
                X = f['X']
            return X, f['y']
    except FileNotFoundError:
        return None


def wrap(func, *args, **kwargs):
    wrapper = lambda: lru_cache()(func)(*args, **kwargs)
    for attr in [attr for attr in dir(func) if not attr.startswith('__')]:
        setattr(wrapper, attr, getattr(func, attr))
    return wrapper
