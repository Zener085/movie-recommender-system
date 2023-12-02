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
    _user = df[df["UserId"] == _user_id][["Age", "Gender", "Occupation"]].iloc[0].tolist()
    _user.append([])
    for _title, _rating in df[df["UserId"] == _user_id][["Title", "Rating"]].values:
        if _rating > 3:
            _user[-1].append((str(_title), int(_rating)))

    return _user


def _generate_test(__data: pd.DataFrame, _total: int = 1000) -> Tuple[List, List]:
    _user_ids = [np.random.randint(1, __data.shape[0]) for _ in range(_total)]
    _users, _likes = [], []
    for _user_id in _user_ids:
        _user = _generate_user(_user_id, _data)
        _likes.append(_user[-1].pop()[0])
        _users.append(_user)
    return _users, _likes


def evaluate(recommender: Recommender, __data: pd.DataFrame):
    _total = 1000
    _pred = []
    _ideal = [True] * _total
    _users, _likes = _generate_test(__data, _total)
    for _user, _like in zip(_users, _likes):
        _movies = zip(*recommender.suggest(*_user))[0]
        _pred.append(_like in _movies)

    _roc, _acc, _f1 = roc_auc_score(_ideal, _pred), accuracy_score(_ideal, _pred), _ndcg_score(_ideal, _pred)

    print(f"ROC AUC score: {_roc}\nAccuracy: {_acc * 100}%\n F1: {_ndcg}")


if __name__ == "__main__":
    data = pd.read_csv("data/test_ratings.csv")
    evaluate(Recommender(5, data), data)
