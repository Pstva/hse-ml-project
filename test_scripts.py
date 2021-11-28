import numpy as np
import random


from src.read_prepare_data import (
    read_cancer_dataset,
    read_spam_dataset,
    train_test_split,
    normalize,
)
from src.metrics import get_precision_recall_accuracy
from src.kdtree import Node, KDTree
from src.knn import KNearest


class TestReadData:
    def test_shapes(self):
        X, y = read_cancer_dataset("data/cancer.csv")
        assert y.shape == (X.shape[0],)
        X, y = read_spam_dataset("data/spam.csv")
        assert y.shape == (X.shape[0],)

    def test_labels(self):
        X, y = read_cancer_dataset("data/cancer.csv")
        assert np.all(np.unique(y) == np.array([0, 1]))
        X, y = read_spam_dataset("data/spam.csv")
        assert np.all(np.unique(y) == np.array([0, 1]))


class TestTrainTestSplit:
    def test_shapes(self):
        for i in range(1, 10):
            train_ratio = i / 10
            n, m = random.randint(10, 10000), random.randint(1, 1000)
            X = np.random.randn(m, n)
            y = np.random.randn(m)
            X_train, y_train, X_test, y_test = train_test_split(X, y, ratio=train_ratio)
            assert X_train.shape[0] == y_train.shape[0]
            assert X_test.shape[0] == y_test.shape[0]
            assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
            assert X_train.shape[0] == int(X.shape[0] * train_ratio)
            assert X_train.shape[1] == X_test.shape[1] == X.shape[1]


class TestNormalizeData:
    def test_values(self):
        X_train = np.random.randn(1000, 10)
        X_test = np.random.randn(100, 10)
        X_train_norm, X_test_norm = normalize(X_train, X_test)
        assert np.all(X_train_norm <= 1)
        assert np.all(X_train_norm >= 0)

    def test_shapes(self):
        X_train = np.random.randn(1000, 10)
        X_test = np.random.randn(100, 10)
        X_train_norm, X_test_norm = normalize(X_train, X_test)
        assert X_train_norm.shape == X_train.shape
        assert X_test_norm.shape == X_test.shape


class TestMetrics:
    def test_accuracy(self):
        y_pred = np.array([1, 1, 1, 1])
        y_true = np.array([1, 1, 1, 1])
        assert get_precision_recall_accuracy(y_pred, y_true)[2] == 1
        y_pred = np.array([0, 0, 0, 1])
        y_true = np.array([1, 1, 1, 1])
        assert get_precision_recall_accuracy(y_pred, y_true)[2] == 0.25
        y_pred = np.array([0, 0, 1, 1])
        y_true = np.array([0, 1, 0, 1])
        assert get_precision_recall_accuracy(y_pred, y_true)[2] == 0.5
        y_pred = np.array([0, 1, 1, 1])
        y_true = np.array([0, 0, 1, 1])
        assert get_precision_recall_accuracy(y_pred, y_true)[2] == 0.75

    def test_precision(self):
        y_pred = np.array([1, 1, 1, 1])
        y_true = np.array([1, 1, 1, 1])
        assert np.all(get_precision_recall_accuracy(y_pred, y_true)[0] == np.array([1]))
        y_pred = np.array([0, 0, 1, 1])
        y_true = np.array([1, 1, 0, 0])
        assert np.all(
            get_precision_recall_accuracy(y_pred, y_true)[0] == np.array([0, 0])
        )
        y_pred = np.array([0, 1, 1])
        y_true = np.array([0, 0, 1])
        assert np.all(
            get_precision_recall_accuracy(y_pred, y_true)[0] == np.array([1, 0.5])
        )

    def test_recall(self):

        y_pred = np.array([1, 1, 1, 1])
        y_true = np.array([1, 1, 1, 1])
        assert np.all(get_precision_recall_accuracy(y_pred, y_true)[1] == np.array([1]))
        y_pred = np.array([0, 0, 1])
        y_true = np.array([0, 1, 1])
        assert np.all(
            get_precision_recall_accuracy(y_pred, y_true)[1] == np.array([1, 0.5])
        )
        y_pred = np.array([0, 0, 1, 1])
        y_true = np.array([0, 0, 1, 1])
        assert np.all(
            get_precision_recall_accuracy(y_pred, y_true)[1] == np.array([1, 1])
        )


class TestTree:
    def true_closest(self, X_train, X_test, k):
        result = []
        for x0 in X_test:
            bests = list(
                sorted(
                    [(i, np.linalg.norm(x - x0)) for i, x in enumerate(X_train)],
                    key=lambda x: x[1],
                )
            )
            bests = [i for i, d in bests]
            result.append(bests[: min(k, len(bests))])
            return result

    def test_shape(self):
        for k in range(1, 101):
            X_train = np.random.randn(100, 3)
            X_test = np.random.randn(10, 3)
            tree = KDTree(X_train, leaf_size=5)
            predicted = tree.query(X_test, k=k)
            true = self.true_closest(X_train, X_test, k=k)
            assert (
                np.sum(
                    np.abs(
                        np.array(np.array(predicted).shape)
                        - np.array(np.array(true).shape)
                    )
                )
                != 0
            )

    def test_neighbors(self):

        for k in range(1, 101):
            X_train = np.random.randn(100, 3)
            X_test = np.random.randn(10, 3)
            tree = KDTree(X_train, leaf_size=5)
            predicted = tree.query(X_test, k=k)
            true = self.true_closest(X_train, X_test, k=k)
            assert (
                sum(
                    [
                        1
                        for row1, row2 in zip(predicted, true)
                        for i1, i2 in zip(row1, row2)
                        if i1 != i2
                    ]
                )
                == 0
            )


class TestKNN:
    def test_predict_shape(self):
        X_train = np.random.randn(100, 3)
        X_test = np.random.randn(10, 3)
        y_train = np.random.randint(0, 2, 100)
        y_test = np.random.randint(0, 2, 10)
        n_classes = 2
        # разное кол-во соседей
        for k in range(1, 10):
            knn = KNearest(k)
            knn.fit(X_train, y_train)
            probas = knn.predict_proba(X_test)
            res = knn.predict(X_test)
            assert len(probas) == y_test.shape[0]
            assert res.shape[0] == y_test.shape[0]

            for pr in probas:
                assert pr.shape[0] == n_classes

        # разный размер листа
        n_neigh = 5
        for leaf_size in range(1, 50, 5):
            knn = KNearest(n_neighbors=n_neigh, leaf_size=leaf_size)
            knn.fit(X_train, y_train)
            probas = knn.predict_proba(X_test)
            res = knn.predict(X_test)
            assert len(probas) == y_test.shape[0]
            assert res.shape[0] == y_test.shape[0]
            for pr in probas:
                assert pr.shape[0] == n_classes

        # разное кол-во классов
        for n_classes in range(3, 10):
            y_train = np.random.randint(0, n_classes, 100)
            y_test = np.random.randint(0, n_classes, 10)
            knn = KNearest()
            knn.fit(X_train, y_train)
            probas = knn.predict_proba(X_test)
            res = knn.predict(X_test)
            assert len(probas) == y_test.shape[0]
            assert res.shape[0] == y_test.shape[0]
            for pr in probas:
                assert pr.shape[0] == n_classes

    def test_probas(self):
        X_train = np.random.randn(100, 3)
        X_test = np.random.randn(10, 3)
        y_train = np.random.randint(0, 2, 100)
        y_test = np.random.randint(0, 2, 10)

        for k in range(1, 11):
            knn = KNearest(n_neighbors=k)
            knn.fit(X_train, y_train)
            probas = knn.predict_proba(X_test)
            for pr in probas:
                assert np.all(pr >= 0)
                assert np.all(pr <= 1)

    def test_labels(self):
        X_train = np.random.randn(100, 3)
        X_test = np.random.randn(10, 3)
        y_train = np.random.randint(0, 2, 100)
        y_test = np.random.randint(0, 2, 10)

        for k in range(1, 11):
            knn = KNearest(n_neighbors=k)
            knn.fit(X_train, y_train)
            res = knn.predict(X_test)
            assert set(res).issubset(set(y_train))

