import pandas
import numpy as np
from typing import Tuple


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    data = pandas.read_csv(path_to_csv, header=0)
    data["label_bin"] = data["label"].apply(lambda x: 1 if x == "M" else 0)
    data = data.sample(frac=1, random_state=20)
    return (
        np.array(data.drop(["label", "label_bin"], axis=1).values.tolist()),
        np.array(data["label_bin"]),
    )


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """
    data = pandas.read_csv(path_to_csv, header=0)
    data = data.sample(frac=1, random_state=20)
    return (
        np.array(data.drop(["label"], axis=1).values.tolist()),
        np.array(data["label"]),
    )


def train_test_split(
    X: np.array, y: np.array, ratio: float
) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    len_train = int(X.shape[0] * ratio)
    X_train, X_test = X[:len_train], X[len_train:]
    y_train, y_test = y[:len_train], y[len_train:]

    return X_train, y_train, X_test, y_test


# min-max scaler
def normalize(X_train, X_test):
    X_min = np.min(X_train, axis=0)
    X_max = np.max(X_train, axis=0)
    X_train_norm = (X_train - X_min) / (X_max - X_min)
    X_test_norm = (X_test - X_min) / (X_max - X_min)
    return X_train_norm, X_test_norm
