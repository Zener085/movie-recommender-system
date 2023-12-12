# movie recommender system

This is a course-based project.
By given data, I tried to implement a recommendation system that can suggest movies by some personal information.

---

## How to use ##

To use the model, you, firstly, might clone the repo.
The model is implemented in [model.py](models/model.py) package.
At the end of this script, write your data and your favorite movies you want to use and se the results after running the
script.
You also may use the [second notebook](notebooks/2.0-model_tunning.ipynb) in the end or in any cell you want.

---

## Dependencies ##

In this project, I used only basic DS libraries
and [bayesian optimization](https://pypi.org/project/bayesian-optimization/).
You can download all of them by this commands:
```
pip install scikit-learn numpy pandas bayesian-optimization
```

---

## References ##
There were used [movielens'](https://movielens.org/) datasets for the training.
