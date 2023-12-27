# Movie Recommender System #
This project is a movie recommendation system based on user personal information. The system utilizes the
`K Nearest Neighbors` algorithm for suggesting movies to users.

---

## How to Use ##
1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/Zener085/movie-recommender-system.git
   ```
2. Navigate to the project directory.
   ```bash
   cd movie-recommender-system
   ```
3. Open the `model.py` file to input your data and favorite movies.
4. Run the script to see the recommendations.
   ```bash
   python model.py
   ```
Additionally, you can explore the second notebook for more interaction.

## Dependencies ##
Make sure to install the required dependencies using the following command:
```bash
pip install scikit-learn numpy pandas bayesian-optimization
```

## References ##
The project utilizes datasets from [MovieLens](https://movielens.org/) for training the recommendation system.

## Evaluation ##
Evaluate the model's performance using the provided `evaluate.py` script. This script generates test data and measures
accuracy and F1 score.
```bash
python evaluate.py
```

## Author ##
The author of this project is [Zener085](https://github.com/Zener085). Feel free to contribute in this project!

## Conclusion ##
The movie recommender system is an effective tool for suggesting movies based on user information. The project provides
a foundation for future enhancements, incorporating additional data and user preferences for even more accurate
recommendations.
