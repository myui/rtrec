# datasets.py

import pandas as pd
import zipfile
import urllib.request
import os

def load_movielens_1m() -> pd.DataFrame:
    """
    Downloads and loads the MovieLens 1M dataset into a DataFrame with columns
    user_id, item_id, tstamp, and rating. If ratings.dat already exists in the
    ./datasets directory, it will use the existing file.

    Returns:
        pd.DataFrame: A DataFrame containing user_id, item_id, rating, and tstamp, sorted by tstamp.
    """
    # URL and paths
    url: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    dataset_path: str = "datasets/ml-1m.zip"
    extracted_folder: str = "datasets/ml-1m"
    ratings_file: str = os.path.join(extracted_folder, "ratings.dat")

    # Check if ratings.dat already exists
    if os.path.exists(ratings_file):
        print("Using existing ratings.dat file.")
    else:
        # Ensure datasets directory exists
        os.makedirs("datasets", exist_ok=True)

        # Download dataset if not already downloaded
        if not os.path.exists(dataset_path):
            print("Downloading MovieLens 1M dataset...")
            urllib.request.urlretrieve(url, dataset_path)
            print("Download complete.")

        # Extract only ratings.dat
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            if "ml-1m/ratings.dat" in zip_ref.namelist():
                print("Extracting ratings.dat file...")
                zip_ref.extract("ml-1m/ratings.dat", "datasets")
                print("Extraction complete.")
            else:
                raise FileNotFoundError("ratings.dat not found in the downloaded zip file.")

    # Load data into DataFrame with correct column names and sort by timestamp
    df: pd.DataFrame = pd.read_csv(
        ratings_file,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "tstamp"]
    )
    df = df.sort_values(by="tstamp")
    return df

def load_movielens_20m() -> pd.DataFrame:
    """
    Downloads and loads the MovieLens 20M dataset into a DataFrame with columns
    user_id, item_id, tstamp, and rating. If ratings.csv already exists in the
    ./datasets directory, it will use the existing file.

    Returns:
        pd.DataFrame: A DataFrame containing user_id, item_id, rating, and tstamp.
    """
    # URL and paths
    url: str = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    dataset_path: str = "datasets/ml-20m.zip"
    extracted_folder: str = "datasets/ml-20m"
    ratings_file: str = os.path.join(extracted_folder, "ratings.csv")

    # Check if ratings.csv already exists
    if os.path.exists(ratings_file):
        print("Using existing ratings.csv file.")
    else:
        # Ensure datasets directory exists
        os.makedirs("datasets", exist_ok=True)

        # Download dataset if not already downloaded
        if not os.path.exists(dataset_path):
            print("Downloading MovieLens 20M dataset...")
            urllib.request.urlretrieve(url, dataset_path)
            print("Download complete.")

        # Extract only ratings.csv
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            if "ml-20m/ratings.csv" in zip_ref.namelist():
                print("Extracting ratings.csv file...")
                zip_ref.extract("ml-20m/ratings.csv", "datasets")
                print("Extraction complete.")
            else:
                raise FileNotFoundError("ratings.csv not found in the downloaded zip file.")

    # Load data into DataFrame with correct column names
    df: pd.DataFrame = pd.read_csv(ratings_file, usecols=["userId", "movieId", "rating", "timestamp"])
    df.columns = ["user_id", "item_id", "rating", "tstamp"]
    df = df.sort_values(by="tstamp")
    return df
