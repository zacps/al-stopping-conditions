import tarfile
import pickle
import glob
from functools import partial

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import fetch_openml
from tabulate import tabulate
from bs4 import BeautifulSoup

# Datasets to implement in order of active learning difficulty (from ALDataset)
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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    dataset = pd.read_csv(url, header=None)
    y = dataset[4].to_numpy()
    isInB = np.array([dataset.to_numpy()[i, 0] > 0.32 for i in range(len(dataset))])
    isInB = isInB.reshape(len(isInB), 1)
    X = dataset.drop([4], axis=1).to_numpy()
    # pca = PCA(n_components=21).fit(X)
    # X = pca.transform(X)
    X = np.append(X, isInB, axis=1)
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)

    return X, y


def bias_banknote(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/haberman's+survival")
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


@source("https://archive.ics.uci.edu/ml/datasets/abalone")
def abalone(dataset_size=1000):
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    )
    dataset = pd.read_csv(url, header=None)
    y = dataset[0].to_numpy()
    isInB = np.array([dataset.to_numpy()[i, 6] < 0.144 for i in range(len(dataset))])
    isInB = isInB.reshape(len(isInB), 1)
    X = dataset.drop([0, 6], axis=1).to_numpy()
    X = np.append(X, isInB, axis=1)
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)

    return X, y


def bias_abalone(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/car+evaluation")
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


def cardio(dataset_size=1000):
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

    return X, y


def bias_cardio(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


def shuttle(dataset_size=1000):
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

    return X, y


def bias_shuttle(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/skin+segmentation")
def skin(dataset_size=1000):
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

    return X, y


def bias_skin(data, labels):
    isInB = data[:, -1]
    X = data[isInB == 1]
    X = X[:, 0 : (len(data[0]) - 1)]
    y = labels[isInB == 1]
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)")
def german():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
    dataset = pd.read_csv(url, header=None, delim_whitespace=True)
    y = dataset[24].to_numpy()
    X = dataset.drop([24], axis=1).to_numpy()
    return X, y


@source("http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)")
def sonar():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    dataset = pd.read_csv(url, header=None)
    y = dataset[60].to_numpy()
    X = dataset.drop([60], axis=1).to_numpy()
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)")
def splice(dataset_size=1000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data"
    dataset = pd.read_csv(url, header=None)
    y = dataset[0].to_numpy()
    X = dataset[2].apply(lambda x: pd.Series(list(x.strip()))).to_numpy()
    X = sklearn.preprocessing.OneHotEncoder().fit_transform(X)
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
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
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\BloodBrainBarrierPenetration\dataset_cddd.csv", header=0)
    X = dataset.iloc[:,2:514].to_numpy()
    y = dataset["penetration"].to_numpy()
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y


@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
def hiv(dataset_size=1000):
    """
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\HIV\dataset_cddd.csv")
    y = dataset["activity"].to_numpy()
    X = dataset.iloc[:,3:515].to_numpy()
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y


@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
def mutagen(dataset_size=1000):
    """
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\Mutagenicity\dataset_cddd.csv")
    y = dataset['mutagen'].to_numpy()
    X = dataset.iloc[:,2:514].to_numpy()
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y


@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
def MUV(dataset_id, dataset_size=1000):
    """
    Contains datasets 466, 548, 600, 644, 652, 689, 692, 712, 713, 733, 737, 810, 832, 846, 852, 858, 859
    
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    assert(dataset_id in [466, 548, 600, 644, 652, 689, 692, 712, 713, 733, 737, 810, 832, 846, 852, 858, 859])
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\MUV\dataset_cddd.csv")
    dataset = dataset[dataset[str(dataset_id)] != '?']
    y = dataset[str(dataset_id)].to_numpy('int')
    X = dataset.iloc[:,dataset.columns.get_loc('cddd_1'):dataset.columns.get_loc('cddd_512')].to_numpy('float')
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y
  
    
@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
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


@source("https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X")
def tox21(dataset_id):
    """
    Contains datasets:
    ['nr-ahr','nr-ar-lbd','nr-aromatase','nr-ar','nr-er-lbd','nr-er','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53']
    
    Performance saturates immediately on a linear svm, not a good active learning target.
    
    https://linkinghub.elsevier.com/retrieve/pii/S001048252030528X
    """
    
    assert(dataset_id in ['nr-ahr','nr-ar-lbd','nr-aromatase','nr-ar','nr-er-lbd','nr-er','nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmp','sr-p53'])
    
    dataset = pd.read_csv(r"F:\Downloads\compound_datasets\collection\tox21\dataset_cddd.csv")
    dataset = dataset[dataset[str(dataset_id)] != '?']
    y = dataset[str(dataset_id)].to_numpy('int')
    X = dataset.iloc[:,dataset.columns.get_loc('cddd_1'):dataset.columns.get_loc('cddd_512')].to_numpy('float')
    
    X, _, y, _ = train_test_split(X, y, train_size=1000, random_state=42)
    
    return X, y


@domain("image")
def mnist(dataset_size=1000, digits=None):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    if digits is not None:
        assert len(digits) == 2
        idx = np.where((y == str(digits[0])) | (y == str(digits[1])))
        X = X[idx]
        y = y[idx]
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y


@domain("image")
@source("https://github.com/googlecreativelab/quickdraw-dataset")
def quickdraw(dataset_size=1000, classes=("cat", "dolphin", "angel", "face")):
    """
    Using the same classes as 'Adversarial Active Learning'.
    """
    
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
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
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
        
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)

    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection")
@domain("nlp")
def reuters21578(dataset_size=1000):
    """
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz'
    dataset = pd.read_csv(url, compression='gzip', header=None)
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return dataset


@domain("nlp")
def rcv1(dataset_size=1000, category='CCAT'):
    bunch = datasets.fetch_rcv1()
    y = bunch.target.getcol(np.where(bunch.target_names == category)[0][0])
    y = np.squeeze(np.array(y.todense()))
    X = bunch.data
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y
    
    
@source("http://www.cs.toronto.edu/~kriz/cifar.html")
@domain("image")
def cifar10(dataset_size=1000):
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
    
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y
        
      
@domain("physics")
def higgs():
    dataset = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz', compression='gz')
    y = dataset[0].to_numpy()
    X = dataset.iloc[:,1:].to_numpy()
    
    X, _, y, _ = train_test_split(X, y, test_size=0.84, random_state=42)
    
    return X, y


@source("http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/")
@domain("nlp")
def webkb(dataset_size=1000):
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
    y = np.array([doc[1] for doc in docs])
    
    X = sklearn.feature_extraction.text.CountVectorizer(min_df=2).fit_transform(X)
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y


@source("https://spamassassin.apache.org/old/publiccorpus/")
@domain("nlp")
def spamassassin(dataset_size=1000):
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
    y = np.array([doc[1] for doc in docs])
    X = sklearn.feature_extraction.text.CountVectorizer(min_df=2).fit_transform(X)
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y


@source("http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions")
def smartphone(dataset_size=1000):
    """
    http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
    """
    
    # Set header=None?
    X_train = pd.read_csv('datasets/smartphone/Train/X_train.txt', sep=' ')
    X_test = pd.read_csv('datasets/smartphone/Test/X_test.txt', sep=' ')
    y_train = pd.read_csv('datasets/smartphone/Train/y_train.txt')
    y_test = pd.read_csv('datasets/smartphone/Test/y_test.txt')
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)

    return (np.concatenate((X_train.to_numpy(), X_test.to_numpy()), axis=0), 
            np.concatenate((y_train.to_numpy(), y_test.to_numpy()), axis=0))


@source("https://archive.ics.uci.edu/ml/datasets/Covertype")
def covertype(dataset_size=1000):
    """
    https://archive.ics.uci.edu/ml/datasets/Covertype
    """
    
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz', header=None)
    y = data.iloc[:,-1]
    X = data.iloc[:,0:-1]
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
    
    return X, y
    
    
@domain("physical")
def htru2(dataset_size=1000):
    data = pd.read_csv('datasets/HTRU2/HTRU_2.csv', header=None)
    y = data.iloc[:,-1]
    X = data.iloc[:,0:-1]
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
        
    return X, y


@domain("computer")
def ida2016(dataset_size=1000):
    test = pd.read_csv('datasets/IDA2016Challenge/aps_failure_test_set.csv', header=14, sep=',')
    train = pd.read_csv('datasets/IDA2016Challenge/aps_failure_training_set.csv', header=14, sep=',')
    
    y = np.concatenate((test.iloc[:,0].to_numpy(), train.iloc[:,0].to_numpy()))
    X = np.concatenate((test.iloc[:,1:].to_numpy(), train.iloc[:,1:].to_numpy()))
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
        
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Detect+Malware+Types")
@domain("computer")
def malware(dataset_size=1000):
    data = pd.read_csv('datasets/malware/staDynVt2955Lab.csv')
    X = data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
        
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Shill+Bidding+Dataset")
def bidding(dataset_size=1000):
    """
    https://archive.ics.uci.edu/ml/datasets/Shill+Bidding+Dataset
    """
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00562/Shill%20Bidding%20Dataset.csv')
    X = data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
        
    return X, y


@source("https://archive.ics.uci.edu/ml/datasets/Swarm+Behaviour")
def swarm(dataset_size=1000, predict='flocking'):
    data = pd.read_csv(f'datasets/swarm/{predict.capitalize()}.csv')
    X = data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    
    if dataset_size is not None:
        X, _, y, _ = train_test_split(X, y, train_size=dataset_size, random_state=42)
        
    return X, y
    