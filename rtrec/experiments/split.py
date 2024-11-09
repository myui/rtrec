import pandas as pd
from typing import Tuple, Optional

def leave_one_last_item(
        df: pd.DataFrame,
        sort_by_tstamp: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a leave-one-last-item split on the dataset.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame with columns ['user', 'item', 'tstamp', 'rating'].

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test DataFrames.
            - train_df: Training data containing all but the last interaction per user.
            - test_df: Test data containing the last interaction per user.
    """
    # Sort the DataFrame by user and timestamp to ensure chronological order
    df = df.sort_values(by=['user', 'tstamp']).reset_index(drop=True)

    # Get the index of each user's last interaction (to be used in the test set)
    last_interaction_idx = df.groupby('user').tail(1).index

    # Split the data into train and test sets
    test_df = df.loc[last_interaction_idx].reset_index(drop=True)
    train_df = df.drop(last_interaction_idx).reset_index(drop=True)

    # Optionally, sort the final train and test DataFrames by timestamp
    if sort_by_tstamp:
        train_df = train_df.sort_values(by='tstamp').reset_index(drop=True)
        test_df = test_df.sort_values(by='tstamp').reset_index(drop=True)

    return train_df, test_df

def temporal_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    timestamp: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a temporal split on the dataset based on either a specified test fraction or timestamp.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame with columns ['user', 'item', 'tstamp', 'rating'].
        test_frac (float): The proportion of data to include in the test set, between 0 and 1. 
                           Used if timestamp is not provided. Default is 0.2.
        timestamp (Optional[float]): The POSIX timestamp threshold for the split. Interactions before this
                                     timestamp will be in the training set, and interactions on or after
                                     this timestamp will be in the test set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test DataFrames.
            - train_df: Training data with interactions before the calculated split point or timestamp.
            - test_df: Test data with interactions on or after the calculated split point or timestamp.
    """
    # Ensure that test_frac is a valid proportion
    assert 0 < test_frac < 1, f"test_frac must be between 0 and 1: {test_frac}"

    # Sort the DataFrame by timestamp to ensure chronological order
    df = df.sort_values(by='tstamp').reset_index(drop=True)

    if timestamp is not None:
        # Split based on the specified timestamp
        train_df = df[df['tstamp'] < timestamp].reset_index(drop=True)
        test_df = df[df['tstamp'] >= timestamp].reset_index(drop=True)
    else:
        # Split based on the specified test fraction
        split_index = int((1 - test_frac) * len(df))
        train_df = df.iloc[:split_index].reset_index(drop=True)
        test_df = df.iloc[split_index:].reset_index(drop=True)

    return train_df, test_df

def temporal_user_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    sort_by_tstamp: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a temporal split on the dataset for each user individually based on a specified test fraction.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame with columns ['user', 'item', 'tstamp', 'rating'].
        test_frac (float): The proportion of each user's data to include in the test set, between 0 and 1.
                           Default is 0.2.
        sort_by_tstamp (bool): Whether to sort the resulting DataFrames by timestamp. Default is True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test DataFrames.
            - train_df: Training data with interactions before the calculated split point for each user.
            - test_df: Test data with interactions on or after the calculated split point for each user.
    """
    # Check that test_frac is a valid proportion
    assert 0 < test_frac < 1, "test_frac must be between 0 and 1."

    # Sort the DataFrame by user and timestamp to ensure chronological order within each user
    df = df.sort_values(by=['user', 'tstamp']).reset_index(drop=True)

    train_data = []
    test_data = []

    # Split data for each user individually
    for user, user_df in df.groupby('user'):
        # Determine the split point based on the test fraction for the user
        split_index = int((1 - test_frac) * len(user_df))

        # Split the user's data into train and test sets
        train_data.append(user_df.iloc[:split_index])
        test_data.append(user_df.iloc[split_index:])

    # Concatenate individual user splits back into DataFrames
    train_df = pd.concat(train_data).reset_index(drop=True)
    test_df = pd.concat(test_data).reset_index(drop=True)

    # Optionally, sort the final train and test DataFrames by timestamp
    if sort_by_tstamp:
        train_df = train_df.sort_values(by='tstamp').reset_index(drop=True)
        test_df = test_df.sort_values(by='tstamp').reset_index(drop=True)

    return train_df, test_df

def random_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a random split on the dataset based on a specified test fraction.

    Parameters:
        df (pd.DataFrame): A pandas DataFrame with columns ['user', 'item', 'tstamp', 'rating'].
        test_frac (float): The proportion of data to include in the test set, between 0 and 1. Default is 0.2.
        random_seed (Optional[int]): The random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test DataFrames.
            - train_df: Training data with a random subset of interactions.
            - test_df: Test data with the remaining interactions.
    """
    # Ensure that test_frac is a valid proportion
    assert 0 < test_frac < 1, f"test_frac must be between 0 and 1: {test_frac}"

    # Randomly shuffle the DataFrame
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Calculate the number of interactions for the test set
    test_size = int(test_frac * len(df))

    # Split the data into train and test sets
    train_df = df.iloc[test_size:].reset_index(drop=True)
    test_df = df.iloc[:test_size].reset_index(drop=True)

    return train_df, test_df
