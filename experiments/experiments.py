import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load datasets
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(df):
    # Example preprocessing: Convert ratings to a range [0, 1]
    df['rating'] = df['rating'] / df['rating'].max()
    return df

# Recommender class (you can replace this with your actual model)
class Recommender:

    def __init__(self, model):
        self.model = model

    def fit(self, train_data):
        # Train the recommender on train_data
        pass

    def predict(self, user_id, item_id):
        # Predict rating for user_id and item_id
        return np.random.rand()  # Placeholder for actual prediction logic

    def evaluate(self, test_data):
        y_true = test_data['rating']
        y_pred = test_data.apply(lambda x: self.predict(x['user_id'], x['item_id']), axis=1)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae

# Datasets paths
datasets = {
    'movielens_1m': 'datasets/movielens_1m.csv',
    'movielens_20m': 'datasets/movielens_20m.csv',
    'epinions': 'datasets/epinions.csv',
    'yelp': 'datasets/yelp.csv',
    'amazon_music': 'datasets/amazon_music.csv',
    'amazon_electronics': 'datasets/amazon_electronics.csv'
}

# Evaluate the recommender on each dataset
results = {}

for name, path in datasets.items():
    print(f"Evaluating recommender on {name}...")
    # Load and preprocess the dataset
    data = load_dataset(path)
    data = preprocess_data(data)

    # Split into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Initialize and train the recommender
    recommender = Recommender()
    recommender.fit(train_data)

    # Evaluate the recommender
    rmse, mae = recommender.evaluate(test_data)
    results[name] = {'RMSE': rmse, 'MAE': mae}

# Print results
for dataset, metrics in results.items():
    print(f"{dataset} - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
