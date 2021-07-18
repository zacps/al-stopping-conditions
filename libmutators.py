import scipy
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from libactive import csr_vappend


def bias(
    X_train,
    X_test,
    y_train,
    y_test,
    amount=1e-1,
    rand=None,
    config_str=None,
    i=None,
    **kwargs,
):
    """
    Bias data. Find the second most predictive attribute and reduce the prevalence of values above the
    mean for the attribute to amount %. Then, remove the attribute from the test and train data.

    This is supposed to simulate the data being biased by an unknown feature.
    """
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X_train[:1000], y_train[:1000])
    classes = tree.predict(X_train)
    u_classes = np.unique(classes, return_counts=True)

    above_idx = np.where(classes == u_classes[0][np.argmax(u_classes[1])])[0]
    above_idx = rand.choice(above_idx, int(above_idx.shape[0] * amount), replace=False)
    below_idx = np.where(classes != u_classes[0][np.argmax(u_classes[1])])[0]

    X_train = X_train[np.concatenate((above_idx, below_idx))]
    y_train = y_train[np.concatenate((above_idx, below_idx))]

    # X_train = np.delete(X_train, second_most_predictive, axis=1)
    # X_test = np.delete(X_test, second_most_predictive, axis=1)

    # TODO: Shuffle!

    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert X_test.shape[0] == X_test.shape[0]

    return X_train, X_test, y_train, y_test


def unbalanced(
    X_train,
    X_test,
    y_train,
    y_test,
    amount=1e-1,
    rand=None,
    config_str=None,
    i=None,
    test_size=None,
    shuffle=None,
    **kwargs,
):
    # Unbalancing might destroy the split, so we undo it and repeat it afterwards
    # Messy, but it should work.
    train_shape = X_train.shape[0]
    test_shape = X_test.shape[0]
    if isinstance(X_train, scipy.sparse.csr_matrix):
        X = csr_vappend(X_train, X_test)
    else:
        X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    unique = np.unique(y)

    idx4 = y == unique[0]
    y4 = y[idx4]
    X4 = X[idx4]
    idx9 = y == unique[1]
    y9 = y[idx9]
    X9 = X[idx9]
    idx = rand.choice(len(y4), int(y4.shape[0] * amount), replace=False)
    X4 = X4[idx]
    y4 = y4[idx]

    if isinstance(X_train, scipy.sparse.csr_matrix):
        X = csr_vappend(X4, X9)
    else:
        X = np.concatenate((X4, X9))
    y = np.concatenate((y4, y9))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rand, shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test


def unbalanced2(
    X_train,
    X_test,
    y_train,
    y_test,
    amount=5e-1,
    rand=None,
    test_size=None,
    shuffle=None,
    **kwargs,
):
    """
    Amount is the fraction that the majority class should take up in the final data. All other classes are reduced to match
    this proportion.
    """

    # Recalculate class proportions
    class_prop = np.unique(y_train, return_counts=True)

    # Second majority class
    majority = rand.choice(class_prop[0][class_prop[1] == class_prop[1].max()])
    n_in_majority_class = class_prop[1][class_prop[0] == majority]

    # Reduce all other classes counts so they make up 1-amount total % of the data,
    # preserving their distribution.
    Xn = [X_train[y_train == majority]]
    yn = [y_train[y_train == majority]]
    for idx, (klass, n) in enumerate(zip(class_prop[0], class_prop[1])):
        if klass == majority:
            continue

        this_class_share = n / np.where(y_train != majority)[0].shape[0]
        n_this_class = int(
            n_in_majority_class * (1 - amount) / amount * this_class_share
        )

        klass = np.where(y_train == klass)[0]
        try:
            new_idx = rand.choice(klass, n_this_class, replace=False)
        except ValueError as e:
            print(f"Tried to pick class {n_this_class} instances from {klass.shape[0]}")
            raise e
        Xn.append(X_train[new_idx])
        yn.append(y_train[new_idx])

    if isinstance(X_train, scipy.sparse.csr_matrix):
        X_train = scipy.sparse.vstack(Xn)
    else:
        X_train = np.concatenate(Xn)
    y_train = np.concatenate(yn)

    # Shuffle train set
    train_idx = rand.choice(y_train.shape[0], y_train.shape[0], replace=False)

    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    return X_train, X_test, y_train, y_test
