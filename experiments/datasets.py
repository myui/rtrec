# datasets.py

import pandas as pd
import zipfile
import urllib.request
import os
from typing import Tuple

def load_movielens(dataset_version: str) -> pd.DataFrame:
    """
    Downloads and loads the specified MovieLens dataset version into a DataFrame with columns
    user_id, item_id, rating, and tstamp. Supported versions are '1m', '20m', and '25m'.

    Args:
        dataset_version (str): The version of the MovieLens dataset to load ('1m', '20m', or '25m').

    Returns:
        pd.DataFrame: A DataFrame containing user_id, item_id, rating, and tstamp.
    """
    # Mapping version to URL and file details
    dataset_info = {
        "1m": {
            "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "zip_path": "datasets/ml-1m.zip",
            "extracted_folder": "datasets/ml-1m",
            "ratings_file": "ml-1m/ratings.dat",
            "sep": "::",
            "columns": ["user_id", "item_id", "rating", "tstamp"]
        },
        "20m": {
            "url": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
            "zip_path": "datasets/ml-20m.zip",
            "extracted_folder": "datasets/ml-20m",
            "ratings_file": "ml-20m/ratings.csv",
            "sep": ",",
            "columns": ["userId", "movieId", "rating", "timestamp"]
        },
        "25m": {
            "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
            "zip_path": "datasets/ml-25m.zip",
            "extracted_folder": "datasets/ml-25m",
            "ratings_file": "ml-25m/ratings.csv",
            "sep": ",",
            "columns": ["userId", "movieId", "rating", "timestamp"]
        }
    }

    # Validate dataset version
    if dataset_version not in dataset_info:
        raise ValueError("Invalid dataset version. Supported versions are '1m', '20m', and '25m'.")

    info = dataset_info[dataset_version]
    ratings_path = os.path.join(info["extracted_folder"], os.path.basename(info["ratings_file"]))

    # Check if the file already exists
    if os.path.exists(ratings_path):
        print(f"Using existing {os.path.basename(ratings_path)} file.")
    else:
        # Ensure datasets directory exists
        os.makedirs("datasets", exist_ok=True)

        # Download dataset if not already downloaded
        if not os.path.exists(info["zip_path"]):
            print(f"Downloading MovieLens {dataset_version} dataset...")
            urllib.request.urlretrieve(info["url"], info["zip_path"])
            print("Download complete.")

        # Extract only the ratings file
        with zipfile.ZipFile(info["zip_path"], "r") as zip_ref:
            if info["ratings_file"] in zip_ref.namelist():
                print(f"Extracting {os.path.basename(info['ratings_file'])} file...")
                zip_ref.extract(info["ratings_file"], "datasets")
                print("Extraction complete.")
            else:
                raise FileNotFoundError(f"{os.path.basename(info['ratings_file'])} not found in the downloaded zip file.")

    # Load data into DataFrame with correct column names and sort by timestamp
    df = pd.read_csv(ratings_path, sep=info["sep"], engine="python", names=info["columns"])
    df.columns = ["user_id", "item_id", "rating", "tstamp"]
    df = df.sort_values(by="tstamp", ascending=True)
    return df

# Specific functions for loading each dataset version
def load_movielens_1m() -> pd.DataFrame:
    return load_movielens("1m")

def load_movielens_20m() -> pd.DataFrame:
    return load_movielens("20m")

def load_movielens_25m() -> pd.DataFrame:
    return load_movielens("25m")
