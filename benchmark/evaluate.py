"""
Evaluation of the model.
"""
__author__ = "Zener085"
__version__ = "1.0.0"
__license__ = "MIT"
__all__ = ["evaluate"]

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from typing import Tuple, List
from models.model import Recommender


def _generate_user(__id: int, __data: pd.DataFrame) -> List:
    _user = __data[__data["UserId"] == __id][["Age", "Gender", "Occupation"]].iloc[0].tolist()
    _user.append([])
    for _title, _rating in __data[__data["UserId"] == __id][["Title", "Rating"]].values:
        if _rating > 3:
            _user[-1].append((str(_title), int(_rating)))

    return _user


def _generate_test(__data: pd.DataFrame, __total: int = 1000) -> Tuple[List, List]:
    _users, _likes = [], []
    while len(_users) < __total:
        _user_id = np.random.choice(__data["UserId"].unique())
        _user = _generate_user(_user_id, __data)
        if not len(_user[-1]):
            continue
        _likes.append(_user[-1].pop()[0])
        _users.append(_user)
    return _users, _likes


def evaluate(recommender: Recommender, __data: pd.DataFrame):
    _total = 1000
    _pred = []
    _ideal = [True] * _total
    _users, _likes = _generate_test(__data, _total)
    for _user, _like in zip(_users, _likes):
        _movies = list(map(lambda x: x[0], recommender.suggest(*_user)))
        _pred.append(_like in _movies)

    _acc, _f1 = accuracy_score(_ideal, _pred), f1_score(_ideal, _pred)

    print(f"Accuracy: {_acc * 100}%\n F1: {_f1}")


if __name__ == "__main__":
    data = pd.read_csv("data/test_ratings.csv")
    evaluate(Recommender(16, data, leaf_size=64), data)
