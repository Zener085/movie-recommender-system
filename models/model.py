"""
Implementation of the Movie Recommender System using K Nearest Neighbors algorithm.
"""
__author__ = "Zener085"
__version__ = "1.0.0"
__license__ = "MIT"
__all__ = ["Recommender"]

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple, Union


class Recommender:
    """
    Recommender system. Based on the data from the "interim/rating.csv" file.
    """

    def __init__(self, __k: int, __data: pd.DataFrame, **kwargs):
        """
        Recommender constructor.

        Parameters:
            __k: Number of neighbors we consider for suggestions.
            __data: Data to be used to suggest.
                    It's considered the data was prepared for the model.
            __encoder: Encoder for the data used to suggest.
        """
        __data = __data.copy()
        self._k = __k
        _knn_data = self._init_preprocess(__data)
        self._knn = NearestNeighbors(n_neighbors=__k, n_jobs=-1, metric="cosine", **kwargs).fit(_knn_data)

    def _init_preprocess(self, __data: pd.DataFrame):
        """
        Preprocesses the initial data.

        Parameters:
            __data: Dataframe with all ratings and users.
        """
        self._encoder = LabelEncoder()
        __data["Gender"] = __data["Gender"] == "M"
        __data["Occupation"] = self._encoder.fit_transform(__data["Occupation"])

        __data["Rating"] = __data["Rating"].apply(lambda x: 5 - x)
        self._data = __data

        _knn_data = __data.pivot_table("Rating", "UserId", "ItemId").fillna(5)
        _knn_data = pd.merge(_knn_data, __data[["UserId", "Age", "Gender", "Occupation"]], on="UserId")
        _knn_data.columns = _knn_data.columns.astype(str)

        self._scaler = StandardScaler()
        return self._scaler.fit_transform(_knn_data.drop("UserId", axis=1))

    def _preprocess_to_suggest(self, __age: int, __gender: str, __occupation: str,
                               __movies: List[Tuple[Union[int, str], int]]) -> pd.DataFrame:
        """
        Preprocesses data to use it for suggestions.

        Parameters:
            __age: Age of the user.
            __gender: Gender of the user.
            __occupation: Occupation of the user.
            __movies: List of movies the user watched and rated.

        Returns:
            A Dataframe with the preprocessed data.
        """
        _movies_ids = []
        _ratings = []
        for _movie in __movies:
            _movie_id = _movie[0]
            if isinstance(_movie_id, str):
                _item = self._data[self._data["Title"] == _movie_id]["ItemId"]
                if len(_item):
                    _movie_id = int(_item.iloc[0])
            _movies_ids.append(_movie_id)
            _ratings.append(5 - _movie[1])

        _user = {str(_id): 5 if _id not in _movies_ids else _ratings.pop(0) for _id in
                 sorted(self._data["ItemId"].unique().tolist())}

        _user["Age"] = [__age]
        _user["Gender"] = [__gender == "M"]
        _user["Occupation"] = [__occupation]
        _user = pd.DataFrame(_user)

        _user["Occupation"] = self._encoder.transform(_user["Occupation"])

        return self._scaler.transform(_user)

    def suggest(self, age: int, gender: str, occupation: str, movies: List[Tuple[Union[int, str], int]]) \
            -> List[Tuple[str, str]]:
        """
        Suggests top k movies based on the provided data.

        Parameters:
            age: Age of the user.
            gender: Gender of the user.
            occupation: Occupation of the user.
            movies: Movies the user rated.
            It must be constructed as a list of tuples with two values in each item:
            the first item is name|id of the movie, and the second one is its rating.

        Returns:
            List of films: its names and the IMDb url.
        """
        _movies = list(map(lambda x: x[0], movies))
        _params = self._preprocess_to_suggest(age, gender, occupation, movies)

        _neighbors = self._knn.kneighbors(_params, return_distance=False)
        _similar = self._data.iloc[_neighbors[0]]
        return list(dict.fromkeys(zip(_similar["Title"], _similar["IMDB_URL"])))[:self._k]


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../data/interim/ratings.csv")
    recommender = Recommender(5, df)

    test_user = [20, "F", "writer", [("Back to the Future (1985)", 5)]]
    print(*recommender.suggest(*test_user), sep="\n")
