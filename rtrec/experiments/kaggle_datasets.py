import pandas as pd
import os
import requests
import zipfile

import pandas as pd
import os
import requests
import zipfile

from .utils import map_hour_to_period

def load_retailrocket():
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
    print(f"Loading dataset from {events_file_path}...")
    events = pd.read_csv(events_file_path, sep=",", encoding="utf-8")

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
    merged_events['transactionid'] = merged_events['transactionid'].astype(str)

    return merged_events
