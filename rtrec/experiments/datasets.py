# datasets.py

import pandas as pd
import zipfile
import urllib.request
import os

from typing import Tuple
from datetime import datetime

def load_movielens(dataset_scale: str, sort_by_tstamp: bool=False) -> pd.DataFrame:
    """
    Downloads and loads the specified MovieLens dataset version into a DataFrame with columns
    user_id, item_id, rating, and tstamp. Supported versions are '1m', '20m', and '25m'.

    Args:
        dataset_scale (str): The version of the MovieLens dataset to load ('1m', '20m', or '25m').
        sort_by_tstamp (bool): Whether to sort the DataFrame by timestamp. Defaults to False.

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
            "header": None, # No header in the file
            "columns": ["user_id", "item_id", "rating", "tstamp"]
        },
        "20m": {
            "url": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
            "zip_path": "datasets/ml-20m.zip",
            "extracted_folder": "datasets/ml-20m",
            "ratings_file": "ml-20m/ratings.csv",
            "sep": ",",
            "header": 0, # the first line is the header
            "columns": ["userId", "movieId", "rating", "timestamp"]
        },
        "25m": {
            "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
            "zip_path": "datasets/ml-25m.zip",
            "extracted_folder": "datasets/ml-25m",
            "ratings_file": "ml-25m/ratings.csv",
            "sep": ",",
            "header": 0, # the first line is the header
            "columns": ["userId", "movieId", "rating", "timestamp"]
        }
    }

    # Validate dataset version
    if dataset_scale not in dataset_info:
        raise ValueError("Invalid dataset version. Supported versions are '1m', '20m', and '25m'.")

    info = dataset_info[dataset_scale]
    ratings_path = os.path.join(info["extracted_folder"], os.path.basename(info["ratings_file"]))

    # Check if the file already exists
    if os.path.exists(ratings_path):
        print(f"Using existing {os.path.basename(ratings_path)} file.")
    else:
        # Ensure datasets directory exists
        os.makedirs("datasets", exist_ok=True)

        # Download dataset if not already downloaded
        if not os.path.exists(info["zip_path"]):
            print(f"Downloading MovieLens {dataset_scale} dataset...")
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
    df = pd.read_csv(ratings_path, sep=info["sep"], engine="python", names=info["columns"], header=info["header"])
    df.columns = ["user", "item", "rating", "tstamp"]
    if sort_by_tstamp:
        df = df.sort_values(by="tstamp", ascending=True)
    return df

# Specific functions for loading each dataset version
def load_movielens_1m() -> pd.DataFrame:
    return load_movielens("1m")

def load_movielens_20m() -> pd.DataFrame:
    return load_movielens("20m")

def load_movielens_25m() -> pd.DataFrame:
    return load_movielens("25m")

def load_yelp_ratings(sort_by_tstamp: bool = False) -> pd.DataFrame:
    """
    Loads the Yelp dataset containing user ratings for businesses.

    Args:
        sort_by_tstamp (bool): Whether to sort the DataFrame by timestamp. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing columns: user, item, rating, and tstamp.
    """

    # Define dataset path
    dataset_path = "datasets/yelp/yelp_academic_dataset_review.json"

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            "Yelp dataset not found. Please download the dataset from "
            "https://www.yelp.com/dataset and place the file in 'datasets/yelp'."
        )

    # Define data types for columns
    dtypes = {
        "user_id": "string",
        "business_id": "string",
        "stars": "float",
        "date": "string"
    }

    # Load dataset in chunks and filter relevant columns
    reader = pd.read_json(dataset_path, lines=True, chunksize=10_000, dtype=dtypes)
    chunks = [chunk[["user_id", "business_id", "stars", "date"]] for chunk in reader]
    df = pd.concat(chunks, ignore_index=True)

    # Rename columns and convert date to timestamp
    df.rename(columns={"user_id": "user", "business_id": "item", "stars": "rating", "date": "tstamp"}, inplace=True)
    df["tstamp"] = df["tstamp"].apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d").timestamp()))

    if sort_by_tstamp:
        df.sort_values(by="tstamp", ascending=True, inplace=True)

    return df

# Define the category mapping dictionary
AMAZON_CATEGORIES = {
    "Fashion": "AMAZON_FASHION",
    "Beauty": "All_Beauty",
    "Appliances": "Appliances",
    "Arts": "Arts_Crafts_and_Sewing",
    "Automotive": "Automotive",
    "Books": "Books",
    "CDs": "CDs_and_Vinyl",
    "Phones": "Cell_Phones_and_Accessories",
    "Clothing": "Clothing_Shoes_and_Jewelry",
    "Music": "Digital_Music",
    "Electronics": "Electronics",
    "GiftCards": "Gift_Cards",
    "Groceries": "Grocery_and_Gourmet_Food",
    "HomeKitchen": "Home_and_Kitchen",
    "IndustrialScientific": "Industrial_and_Scientific",
    "Kindle": "Kindle_Store",
    "LuxuryBeauty": "Luxury_Beauty",
    "Magazines": "Magazine_Subscriptions",
    "Movies": "Movies_and_TV",
    "MusicalInstruments": "Musical_Instruments",
    "Office": "Office_Products",
    "PatioGarden": "Patio_Lawn_and_Garden",
    "Pets": "Pet_Supplies",
    "PrimePantry": "Prime_Pantry",
    "Software": "Software",
    "Sports": "Sports_and_Outdoors",
    "Tools": "Tools_and_Home_Improvement",
    "Toys": "Toys_and_Games",
    "VideoGames": "Video_Games",
}

def load_amazon_review_v2(category_name: str = "Music",
                          small_subsets: bool = True,
                          parse_image_url: bool = False,
                          sort_by_tstamp: bool = False
) -> pd.DataFrame:
    """
    Downloads and loads the Amazon review dataset for a specified category (or all categories)
    and returns it as a pandas DataFrame. Can also load the full dataset or the 5-core subset.

    Args:
        category_name (str): Category of the Amazon dataset to load. Defaults to "Music".
                        Use "all" to load all available categories.
        small_subsets (bool): Whether to load the 5-core subset (default is True).
        parse_image_url (bool): Whether to parse image URLs from the dataset (default is False).
        sort_by_tstamp (bool): Whether to sort the DataFrame by timestamp (default is False).

    Returns:
        pd.DataFrame: A DataFrame containing user, item, rating, timestamp, image_url, and 
                      (if category is "all") category.
    """
    # Base URL and directory setup
    base_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/"
    data_dir = "datasets/amazon_v2"
    os.makedirs(data_dir, exist_ok=True)

    final_df = pd.DataFrame()
    categories = AMAZON_CATEGORIES.keys() if category_name == "all" else [category_name]

    for category in categories:
        # Translate the category name to match file naming
        category_key = AMAZON_CATEGORIES.get(category, None)
        if category_key is None:
            print(f"Category '{category}' not found in the Amazon dataset.")
            continue
        suffix = "_5.json.gz" if small_subsets else ".json.gz"  # Fix applied here
        file_name = f"{category_key}{suffix}"
        url = f"{base_url}{file_name}"
        local_path = os.path.join(data_dir, file_name)

        if not os.path.exists(local_path):
            print(f"Downloading Amazon {category} dataset...")
            urllib.request.urlretrieve(url, local_path)
            print("Download completed.")

        # Load the dataset for the specified category
        df = pd.read_json(local_path, lines=True)

        # Filter columns and rename them
        df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime', 'image']]
        df.rename(columns={
            'reviewerID': 'user',
            'asin': 'item',
            'overall': 'rating',
            'unixReviewTime': 'timestamp',
            'image': 'image_url',
        }, inplace=True)

        # Replace missing image URLs with the first image in the list, or the default if missing
        if parse_image_url:
            df['image_url'] = df['image_url'].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else
                        f"http://images.amazon.com/images/P/{df['item']}.01._SCTZZZZZZZ_.jpg"
            )
        else:
            df.drop(columns=['image_url'], inplace=True)

        # Add the category name if we are processing multiple categories
        if category_name == 'all':
            df['category'] = category

        # Concatenate the new dataframe to the final one
        final_df = pd.concat([final_df, df], ignore_index=True)
        print(f"{category} dataset loaded successfully.")

    if sort_by_tstamp:
        final_df.sort_values(by="timestamp", ascending=True, inplace=True)

    return final_df

def load_amazon_music_v2(small_subsets: bool = True, parse_image_url: bool = False, sort_by_tstamp: bool = False) -> pd.DataFrame:
    return load_amazon_review_v2("Music", small_subsets, parse_image_url, sort_by_tstamp)

def load_amazon_electronics_v2(small_subsets: bool = True, parse_image_url: bool = False, sort_by_tstamp: bool = False) -> pd.DataFrame:
    return load_amazon_review_v2("Electronics", small_subsets, parse_image_url, sort_by_tstamp)

def load_dataset(name: str) -> pd.DataFrame:
    """
    Loads a dataset by name.

    Args:
        name (str): Name of the dataset to load.

    Returns:
        pd.DataFrame: The loaded dataset as a DataFrame.
    """
    if name == "movielens_1m":
        return load_movielens_1m()
    elif name == "movielens_20m":
        return load_movielens_20m()
    elif name == "movielens_25m":
        return load_movielens_25m()
    elif name == "yelp":
        return load_yelp_ratings()
    elif name == "amazon_music":
        return load_amazon_music_v2()
    elif name == "amazon_electronics":
        return load_amazon_electronics_v2()
    else:
        raise ValueError(f"Dataset '{name}' not found.")