import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.array(feature_vector, dtype=np.float64)
    target_vector = np.array(target_vector, dtype=int)

    sort_indices = np.argsort(feature_vector)
    counter = Counter(feature_vector)

    a1 = sorted(counter.items(), key=lambda x: x[0])
    a2 = list(map(lambda x: x[1], a1))
    Rl = np.cumsum(a2)[:-1]

    unique_feature_vector, unique_indices = np.unique(feature_vector[sort_indices], return_index=True)
    target_vector = target_vector[sort_indices]

    thresholds = 0.5 * (unique_feature_vector[1:] + unique_feature_vector[:-1])

    cum_indices = (unique_indices-1)[1:]
    cum_target = np.cumsum(target_vector)[cum_indices]

    p1 = cum_target / Rl
    Hl = 1 - p1**2 - (1 - p1)**2

    cum_indices2 = (len(target_vector) - unique_indices - 1)[1:]
    cum_target2 = np.cumsum(target_vector[::-1])[cum_indices2]

    Rr = len(target_vector) - Rl
    p1 = cum_target2 / Rr
    Hr = 1 - p1**2 - (1 - p1)**2

    ginis = - Rl / len(target_vector) * Hl - Rr / len(target_vector) * Hr

    best_index = np.argmax(ginis)
    return np.array(thresholds), np.array(ginis), thresholds[best_index], ginis[best_index]


# feature_vector = np.array([0.5, 0.7, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1, 0.7])
# target_vector = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
#
# thresholds, ginis, threshold_best, gini_best = find_best_split(feature_vector, target_vector)

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count

                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(set(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            most_common = Counter(sub_y).most_common(1)
            node["class"] = most_common[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        sub_X1 = sub_X[split]
        sub_y1 = sub_y[split]

        sub_X1not = sub_X[np.logical_not(split)]
        self._fit_node(sub_X1, sub_y1, node["left_child"])
        self._fit_node(sub_X1not, sub_y1, node["right_child"])

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        feature = node['feature_split']

        x1 = x[feature]

        if self._feature_types[feature] == 'categorical':
            if np.isin(x1, node['categories_split']):
                return self._predict_node(x, node['left_child'])
            return self._predict_node(x, node['right_child'])
        else:
            if x1 < node['threshold']:
                return self._predict_node(x, node['left_child'])
            return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
