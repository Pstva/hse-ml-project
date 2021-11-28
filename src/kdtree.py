import numpy as np
from typing import Tuple, List


class Node:
    def __init__(
        self, X: np.array, leaf_size: int = 40, parent=None, X_index=None, start_feat=0
    ):

        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области, 
            в которых не меньше leaf_size точек).
        parent : Node
            родитель вершины
        X_index : list
            индексы точек, находящихся в вершине, относительно всего набора точек
        start_feat : int
            индекс признака, начиная с которого пробуем разделить лист

        Returns
        -------

        """

        self.X = X  # векторы всех признаков всех точек
        if X_index is None:
            self.X_index = np.array([i for i in range(self.X.shape[0])])
        else:
            self.X_index = X_index  # индексы точек в листе
        self.leaf_size = leaf_size  # минимально возможный размер листа
        self.left = None  # левый ребенок
        self.right = None  # правый ребенок
        self.parent = parent  # родитель вершины
        self.delimiter = (
            -1,
            -1,
        )  # (feature, median) - по какому признаку делим сам лист
        self.start_feat = start_feat  # начиная с какого признака пробуем делить лист

        # если лист больше, чем минимально возможный размер листа, пробуем разделить лист
        if len(self.X_index) > self.leaf_size:
            self.cut_node()

    def cut_node(self):

        # прямой порядок + (если пройдем до конца признаки) по которым уже делили
        features_order = [f for f in range(self.start_feat, self.X.shape[1])]
        features_order.extend([f for f in range(self.start_feat)])

        # идем по всем признакам
        for f in features_order:
            mid = np.median(self.X[self.X_index, f])
            # выбираем индексы (во  всей матрице признаков) точек, которые потенциально пойдут в детей
            child_left_ind = self.X_index[np.where(self.X[self.X_index, f] <= mid)]
            child_right_ind = self.X_index[np.where(self.X[self.X_index, f] > mid)]

            # нашли признак, по которому можно разделить лист
            if (
                len(child_left_ind) >= self.leaf_size
                and len(child_right_ind) >= self.leaf_size
            ):
                self.delimiter = (f, mid)

                # если это был последний признак по списку, дети начнут делиться с первого признака
                # иначе - со следующего
                if f == self.X.shape[1] - 1:
                    child_start_feat = 0
                else:
                    child_start_feat = f + 1

                self.left = Node(
                    self.X,
                    self.leaf_size,
                    parent=self,
                    X_index=child_left_ind,
                    start_feat=child_start_feat,
                )
                self.right = Node(
                    self.X,
                    self.leaf_size,
                    parent=self,
                    X_index=child_right_ind,
                    start_feat=child_start_feat,
                )
                break

    # соединяет 2 вектора по k соседей в один вектор из k ближайших
    def merge_k(
        self,
        k: int,
        dists: np.array,
        dists_index: np.array,
        opposite_dists: np.array,
        opposite_dists_index: np.array,
    ) -> Tuple[np.array, np.array]:
        i, j = 0, 0
        merged_list = []
        index_merged = []
        while i + j < k and i < len(dists) and j < len(opposite_dists):
            if dists[i] <= opposite_dists[j]:
                merged_list.append(dists[i])
                index_merged.append(dists_index[i])
                i += 1
            else:
                merged_list.append(opposite_dists[j])
                index_merged.append(opposite_dists_index[j])
                j += 1

        while i + j < k and i < len(dists):
            merged_list.append(dists[i])
            index_merged.append(dists_index[i])
            i += 1

        while i + j < k and j < len(opposite_dists):
            merged_list.append(opposite_dists[j])
            index_merged.append(opposite_dists_index[j])
            j += 1

        return np.array(merged_list), np.array(index_merged)

    def node_query(self, x: np.array, k: int) -> Tuple[np.array, np.array]:
        cur_node = self

        # если вершина - лист, то просто добавляем k соседей из нее
        if cur_node.left is None:

            # расстояния от x до всех точек в текущем листе
            dists = np.sqrt(np.sum((cur_node.X[cur_node.X_index] - x) ** 2, axis=1))
            index = np.argsort(dists)[:k]
            dists_index = cur_node.X_index[index]
            return dists[index], dists_index

        else:
            f, mid = cur_node.delimiter
            if x[f] <= mid:
                dists, dists_index = cur_node.left.node_query(x, k)
            else:
                dists, dists_index = cur_node.right.node_query(x, k)

            # идем в соседний лист
            if dists.shape[0] < k or dists[-1] >= abs(x[f] - mid):
                if x[f] <= mid:
                    opposite_dists, opposite_dists_index = cur_node.right.node_query(
                        x, k
                    )
                else:
                    opposite_dists, opposite_dists_index = cur_node.left.node_query(
                        x, k
                    )

                dists, dists_index = self.merge_k(
                    k, dists, dists_index, opposite_dists, opposite_dists_index
                )
            return dists, dists_index


class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области, 
            в которых не меньше leaf_size точек).

        Returns
        -------

        """

        self.X = X
        self.leaf_size = leaf_size
        self.tree = Node(
            self.X, self.leaf_size, parent=None, X_index=None, start_feat=0
        )

    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k): 
            индексы k ближайших соседей для всех точек из X.

        """
        res = []
        for x in X:
            ans, k_closest = self.tree.node_query(x, k)
            res.append(k_closest.tolist())
        return res
