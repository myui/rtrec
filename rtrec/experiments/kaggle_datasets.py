from collections import defaultdict
from typing import Dict
import pandas as pd
import os
import requests
import zipfile
from tqdm import tqdm

from .utils import map_hour_to_period

def load_retailrocket(standardize_schema: bool=True) -> pd.DataFrame:
    """
    Load the Retail Rocket dataset from a CSV file.

    Returns:
        DataFrame: A pandas DataFrame containing the Retail Rocket dataset.
    """
    url = "https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset"
    data_dir = "datasets/kaggle/retailrocket"
    events_file_path = os.path.join(data_dir, "events.csv")

    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download the dataset if it doesn't exist
    if not os.path.exists(events_file_path):
        archive_file = os.path.join(data_dir, "retailrocket.zip")
        if not os.path.exists(archive_file):
            print(f"Downloading dataset from {url}...")
            response = requests.get(url)
            with open(archive_file, "wb") as f:
                f.write(response.content)
            print("Download complete.")

        # extract the dataset
        print(f"Extracting dataset to {data_dir}...")
        with zipfile.ZipFile(archive_file, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

    # Load the dataset
    print(f"Loading dataset from {events_file_path}...!")
    events = pd.read_csv(events_file_path, sep=",", encoding="utf-8", dtype={
        'timestamp': 'int64',
        'visitorid': 'int64',
        'event': 'category', # 'view', 'addtocart', 'transaction'
        'itemid': 'int64',
        'transactionid': 'Int64'  # NaN for non-transaction events
    }, na_values=[""], keep_default_na=True)

    item_props1 = pd.read_csv(os.path.join(data_dir, "item_properties_part1.csv"), sep=",", encoding="utf-8")
    item_props2 = pd.read_csv(os.path.join(data_dir, "item_properties_part2.csv"), sep=",", encoding="utf-8")
    item_properties = pd.concat([item_props1, item_props2])

    events = events.sort_values(['timestamp', 'itemid']).reset_index(drop=True)
    item_properties = item_properties.sort_values(['timestamp', 'itemid']).reset_index(drop=True)
    merged_events = pd.merge_asof(
        events,
        item_properties,
        on='timestamp',
        by='itemid',
        direction='backward', # item_properties.timestamp<=events.timestamp
        allow_exact_matches=True
    )
    merged_events['available'] = merged_events[merged_events['property'] == 'available']['value'].copy()
    merged_events['categoryid'] = merged_events[merged_events['property'] == 'categoryid']['value'].copy()

    category_tree = pd.read_csv(os.path.join(data_dir, "category_tree.csv"), sep=",", encoding="utf-8", dtype=str)
    merged_events = merged_events.merge(category_tree, on='categoryid', how='left')

    def expand_tags(row):
        prop = row['property']
        if pd.isna(prop):
            return []
        if prop in ['available', 'categoryid']:
            return []
        values = str(row['value']).split()
        return [f"{prop}#{v}" for v in values if not v.startswith('n')]

    merged_events['tags'] = merged_events[['property', 'value']].apply(expand_tags, axis='columns')
    merged_events.drop('value', axis='columns', inplace=True)

    date_time = pd.to_datetime(events['timestamp'], unit='ms')
    merged_events['date_time'] = date_time
    merged_events['day_of_week'] = date_time.dt.dayofweek
    merged_events['hour'] = date_time.dt.hour
    merged_events['time_of_day'] = date_time.dt.hour.apply(map_hour_to_period)

    if standardize_schema:
        merged_events.rename(columns={
            'visitorid': 'user',
            'itemid': 'item',
            'timestamp': 'tstamp',
        }, inplace=True)
        # change view to 1, addtocart to 3, transaction to 5
        merged_events['rating'] = merged_events['event'].cat.rename_categories({
            'view': 1,
            'addtocart': 3,
            'transaction': 5
        })

    return merged_events

def load_hm_dataset(
    transactions_url: str = "https://repo.hops.works/dev/jdowling/h-and-m/transactions_train.csv",
    standardize_schema: bool = True) -> pd.DataFrame:
    """
    Load transactions_train.csv and return it as a DataFrame.
    """
    # Directory for saving the data
    data_dir = "datasets/kaggle/h-and-m"
    os.makedirs(data_dir, exist_ok=True)

    # Download the file if it doesn't exist
    transactions_file = os.path.join(data_dir, "transactions_train.csv")
    download_file(transactions_url, transactions_file)

    # Load transactions_train.csv into a DataFrame
    transactions = pd.read_csv(transactions_file, dtype={
        # 't_dat': 'str',
        'customer_id': 'str',
        'article_id': 'Int32',
        'price': 'float32',
        'sales_channel_id': 'Int8'
    }, parse_dates=['t_dat'], date_format='%Y-%m-%d', engine='pyarrow')

    # Note only the last 16 HEX digits are needed to identify unique customers.
    transactions['customer_id'] = transactions['customer_id'].apply(customer_hex_to_int64).astype('int64')

    if standardize_schema:
        transactions.rename(columns={
            'customer_id': 'user',
            'article_id': 'item',
        }, inplace=True)
        # Convert the date to unix timestamp in float64
        # transactions['tstamp'] = transactions['t_dat'].map(pd.Timestamp.timestamp)
        transactions['tstamp'] = transactions['t_dat'].astype('int64') / 10**9
        transactions['rating'] = 1.0
        transactions.drop(columns=['t_dat'], inplace=True)

    return transactions

# Load user tags based on customer features
def load_hm_user_tags(customers_url: str = "https://repo.hops.works/dev/jdowling/h-and-m/customers.csv") -> Dict[int, list]:
    """
    Load user tags based on customer features from a CSV file.

    Args:
        customers_url (str): URL to download the customers dataset.

    Returns:
        dict: A dictionary where keys are customer IDs and values are lists of user tags.
    """
    data_dir = "datasets/kaggle/h-and-m"
    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download customers.csv if not already downloaded
    customers_csv = os.path.join(data_dir, "customers.csv")
    download_file(customers_url, customers_csv)

    # Load the customers data into a DataFrame
    # customer_id,FN,Active,club_member_status,fashion_news_frequency,age,postal_code
    customers = pd.read_csv(customers_csv, dtype={
        'customer_id': 'str',
        'FN': 'Int8',
        'Active': 'Int8',
        'club_member_status': 'category',
        'fashion_news_frequency': 'category',
        'age': 'Int8',
        'postal_code': 'str'
    }, na_values=[""], keep_default_na=False, engine='pyarrow')

    # Note only the last 16 HEX digits are needed to identify unique customers.
    customers['customer_id'] = customers['customer_id'].apply(customer_hex_to_int64).astype('int64')

    # Initialize the dictionary to store user tags
    user_tags = {}
    for _, row in customers.iterrows():
        cid = row["customer_id"]
        tags = []

        if pd.isna(row["FN"]):
            tags.append("FN#0")
        else:
            tags.append(f"FN#{int(row['FN'])}")

        if pd.isna(row["Active"]):
            tags.append("Active#0")
        else:
            tags.append(f"Active#{int(row['Active'])}")

        # Active, PRE-CREATE, OTHER
        tags.append(f"club_member#{row['club_member_status']}")

        # NONE, Regularly, OTHER
        tags.append(f"news_freq#{row['fashion_news_frequency']}")

        if pd.isna(row["age"]):
            tags.append("age#unknown")
        else:
            tags.append(f"age#{bin_age(row['age'])}")

        tags.append(f"postal_code#{row['postal_code']}")

        user_tags[cid] = tags

    return user_tags


# Load item tags based on article features
def load_hm_item_tags(articles_url: str = "https://repo.hops.works/dev/jdowling/h-and-m/articles.csv", include_text_features: bool=False) -> Dict[int, list]:
    """
    Load item tags based on article features from a CSV file.

    Args:
        articles_url (str): URL to download the articles dataset.
        include_text_features (bool): Whether to include descriptive text features like names and descriptions.

    Returns:
        dict: A dictionary where keys are article IDs and values are lists of item tags.
    """
    data_dir = "datasets/kaggle/h-and-m"
    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download articles.csv if not already downloaded
    articles_csv = os.path.join(data_dir, "articles.csv")
    download_file(articles_url, articles_csv)

    # Load the articles data into a DataFrame
    # article_id,
    # product_code,prod_name,product_type_no,product_type_name,product_group_name,
    # graphical_appearance_no,graphical_appearance_name,colour_group_code,colour_group_name,perceived_colour_value_id,perceived_colour_value_name,perceived_colour_master_id,perceived_colour_master_name,
    # department_no,department_name,index_code,index_name,index_group_no,index_group_name,
    # section_no,section_name,garment_group_no,garment_group_name,detail_desc
    articles = pd.read_csv(articles_csv, dtype={
        'article_id': 'Int32',
        'product_code': 'Int32',
        'prod_name': 'str',
        'product_type_no': 'Int16',
        'product_type_name': 'str',
        'product_group_name': 'str',
        'graphical_appearance_no': 'Int32',
        'graphical_appearance_name': 'str',
        'colour_group_code': 'Int8',
        'colour_group_name': 'str',
        'perceived_colour_value_id': 'Int8',
        'perceived_colour_value_name': 'str',
        'perceived_colour_master_id': 'Int8',
        'perceived_colour_master_name': 'str',
        'department_no': 'Int16',
        'department_name': 'str',
        'index_code': 'str',
        'index_name': 'str',
        'index_group_no': 'Int8',
        'index_group_name': 'str',
        'section_no': 'Int8',
        'section_name': 'str',
        'garment_group_no': 'Int16',
        'garment_group_name': 'str',
        'detail_desc': 'str'
    }, na_values=[""], keep_default_na=False, engine='pyarrow')

    # Initialize the dictionary to store item tags
    item_tags = {}
    for _, row in articles.iterrows():
        aid = row["article_id"]
        tags = []

        # Product features
        tags.append(f"product_code#{row['product_code']}")
        tags.append(f"product_type#{row['product_type_no']}")
        tags.append(f"product_group#{row['product_group_name']}")
        if include_text_features:
            tags.append(f"product_name#{row['prod_name']}")
            tags.append(f"product_type_name#{row['product_type_name']}")

        # Graphical appearance
        tags.append(f"graphical_appearance#{row['graphical_appearance_no']}")
        tags.append(f"colour_group#{row['colour_group_code']}")
        tags.append(f"perceived_colour_value#{row['perceived_colour_value_id']}")
        tags.append(f"perceived_colour_master#{row['perceived_colour_master_id']}")
        if include_text_features:
            tags.append(f"graphical_appearance_name#{row['graphical_appearance_name']}")
            tags.append(f"colour_group_name#{row['colour_group_name']}")
            tags.append(f"perceived_colour_value_name#{row['perceived_colour_value_name']}")
            tags.append(f"perceived_colour_master_name#{row['perceived_colour_master_name']}")

        # Department features
        tags.append(f"department#{row['department_no']}")
        if include_text_features:
            tags.append(f"department_name#{row['department_name']}")

        # Index features
        tags.append(f"index#{row['index_code']}")
        tags.append(f"index_group#{row['index_group_no']}")
        if include_text_features:
            tags.append(f"index_name#{row['index_name']}")
            tags.append(f"index_group_name#{row['index_group_name']}")

        # Section features
        tags.append(f"section#{row['section_no']}")
        if include_text_features:
            tags.append(f"section_name#{row['section_name']}")

        # Garment group features
        tags.append(f"garment_group#{row['garment_group_no']}")
        if include_text_features:
            tags.append(f"garment_group_name#{row['garment_group_name']}")

        # Detail description
        if include_text_features:
            tags.append(f"detail_desc#{row['detail_desc']}")

        item_tags[aid] = tags

    return item_tags

def download_file(url: str, file_path: str, chunk_size: int = 8192) -> None:
    """Download a file from a URL and save it to the specified path."""
    if not os.path.exists(file_path):
        print(f"Downloading file from {url} to {file_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # raise an error for bad responses
        with open(file_path, "wb") as file:
            total_length = response.headers.get('content-length')
            if total_length is None:
                file.write(response.content)
            else:
                pbar = tqdm(total=int(total_length), unit="B", unit_scale=True)
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)
                        pbar.update(len(chunk))
                pbar.close()
        print(f"Download complete. File saved to {file_path}.")
    else:
        print(f"File already exists at {file_path}.")

# Binning functions for quantitative variables
def bin_age(age: float) -> str:
    if age < 0:
        return "invalid"
    elif age < 20:
        return "<20"
    elif age < 40:
        return "20-39"
    elif age < 60:
        return "40-59"
    else:
        return "60+"

def customer_hex_to_int64(hex_string) -> int:
    # Convert the last 16 hex digits to an integer
    val = int(hex_string[-16:], 16)
    val &= (1 << 63) - 1  # 0x7FFFFFFFFFFFFFFF
    return val
