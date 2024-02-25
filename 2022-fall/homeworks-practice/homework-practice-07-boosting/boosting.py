from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib as plt


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        model = self.base_model_class(**self.base_model_params)

        bootstrap_indices = np.random.choice(x.shape[0], int(x.shape[0] * self.subsample))
        s = -self.loss_derivative(y, predictions)

        X_boot, y_boot, s_boot = x[bootstrap_indices], y[bootstrap_indices], s[bootstrap_indices]

        model.fit(X_boot, s_boot)  # обучаемся на сдвиги. То есть обучаемся на градиент
        new_predictions = model.predict(x)

        best_gamma = self.find_optimal_gamma(y, predictions, new_predictions)
        self.gammas.append(best_gamma)
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        bad_quality_count = 0
        best_score = 0

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)

            self.history['train_loss'] += [self.loss_fn(train_predictions, y_train)]
            self.history['val_loss'] += [self.loss_fn(valid_predictions, y_valid)]

            if self.early_stopping_rounds is not None:
                score = self.score(x_valid, y_valid)
                if score > best_score:
                    best_score = score
                    bad_quality_count = 0
                else:
                    bad_quality_count += 1
                if bad_quality_count >= self.early_stopping_rounds:
                    break

        if self.plot:
            for i, (loss_name, loss_history) in enumerate(self.history.items()):
                plt.subplot(1, 2, i+1)
                plt.plot(np.arange(self.n_estimators), loss_history)
                plt.xlabel('n_estimators')
                plt.ylabel('loss')
                plt.title(loss_name)
            plt.show()

    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += gamma * model.predict(x)
        result = np.zeros((x.shape[0], 2))
        sigmoids = self.sigmoid(predictions)
        result[:, 1] = sigmoids
        result[:, 0] = 1 - sigmoids
        return result

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        importances = sum([tree.feature_importances_ for tree in self.models])
        importances /= len(self.models)
        importances /= sum(importances)
        return importances
