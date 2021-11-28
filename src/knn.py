import numpy as np
from typing import NoReturn, List
from src.kdtree import KDTree


class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.tree = None
        self.labels = None
        self.classes = None

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.tree = KDTree(X, self.leaf_size)
        self.labels = y
        self.classes = np.unique(y)

    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """
        probas = []
        predictions = self.tree.query(X, self.n_neighbors)
        for pred in predictions:
            res = []
            pred_labels = self.labels[pred]
            for cl in self.classes:
                res.append(len(pred_labels[pred_labels == cl]) / len(pred))
            probas.append(np.array(res))
        return probas

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        return np.argmax(self.predict_proba(X), axis=1)
