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

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
          ]


def _convert_date(__x: str):
    return pd.to_datetime(__x).date().toordinal()


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
        self._init_preprocess(__data)
        _knn_data = self._data.drop(["UserId", "ItemId", "Title", "IMDB_URL"], axis=1)
        self._knn = NearestNeighbors(n_neighbors=300, n_jobs=-1, **kwargs).fit(_knn_data)

    def _init_preprocess(self, __data: pd.DataFrame):
        """
        Preprocesses the initial data.

        Parameters:
            __data: Dataframe with all ratings and users.
        """
        self._encoder = LabelEncoder()
        __data["Gender"] = __data["Gender"] == "M"
        __data["Occupation"] = self._encoder.fit_transform(__data["Occupation"])

        __data["ReleaseDate"] = __data["ReleaseDate"].apply(_convert_date)
        __data["Rating"] = __data["Rating"].apply(lambda x: 5 - x)

        self._scaler = StandardScaler()
        __data[["Rating", "Age", "Gender", "Occupation", "ReleaseDate"]] = self._scaler.fit_transform(
            __data[["Rating", "Age", "Gender", "Occupation", "ReleaseDate"]]
        )

        self._data = __data

    def _preprocess_to_suggest(self, __age: int, __gender: str, __occupation: str,
                               __movies: List[Tuple[Union[int, str], int]]) -> List[pd.DataFrame]:
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
        global GENRES

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

        _users = []

        for _rating, _movie_id in zip(_ratings, _movies_ids):
            _user = pd.DataFrame({
                "ItemId": [_movie_id],
                "Rating": [_rating],
                "Age": [__age],
                "Gender": [__gender == "M"],
                "Occupation": [__occupation]
            })
            _user["Occupation"] = self._encoder.transform(_user["Occupation"])
            _movies_genres = self._data[self._data["ItemId"].isin(_movies_ids)][["ItemId", "ReleaseDate"] + GENRES]

            _user = pd.merge(_user, _movies_genres, on="ItemId").drop("ItemId", axis=1)
            _user[["Rating", "Age", "Gender", "Occupation", "ReleaseDate"]] = self._scaler.transform(
                _user[["Rating", "Age", "Gender", "Occupation", "ReleaseDate"]]
            )
            _users.append(_user)

        return _users

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
        _params_list = self._preprocess_to_suggest(age, gender, occupation, movies)

        _distances, _neighbors = [], []
        for _params in _params_list:
            _distance, _neighbor = self._knn.kneighbors(_params)
            _distances.extend(_distance.tolist()), _neighbors.extend(_neighbor.tolist())
        _, _sorted_neighbors = zip(*sorted(zip(_distances, _neighbors)))
        _similar = self._data.iloc[_sorted_neighbors[0]]
        return list(dict.fromkeys(zip(_similar["Title"], _similar["IMDB_URL"])))[:self._k]


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../data/interim/ratings.csv")
    recommender = Recommender(5, df)

    test_user = [20, "F", "writer", [("Back to the Future (1985)", 5)]]
    print(*recommender.suggest(*test_user), sep="\n")
