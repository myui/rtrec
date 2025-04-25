import pandas as pd
from typing import List

def n_core_filter(df: pd.DataFrame, columns: List[str], min_count: int = 10) -> pd.DataFrame:
    """
    Filters the DataFrame to only include rows where any specified column's values
    appear at least `min_count` times.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (List[str]): The list of columns to apply the n-core filter on.
        min_count (int): Minimum occurrences required for each value in any specified column 
                         to be retained. Defaults to 10.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows where values in any specified 
                      column appear at least `min_count` times.
    """
    # Initialize a mask to keep track of rows that meet the criteria for any column
    mask = pd.Series([False] * len(df))

    # Update the mask for each column
    for column in columns:
        counts = df[column].value_counts()
        valid_values = counts[counts >= min_count].index
        mask |= df[column].isin(valid_values)  # Logical OR to combine conditions

    # Filter the DataFrame based on the accumulated mask
    return df[mask]

def map_hour_to_period(hour: int) -> str:
    if 0 <= hour < 4:
        return 'midnight'
    elif 4 <= hour < 8:
        return 'early morning'
    elif 8 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 14:
        return 'noon'
    elif 14 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 20:
        return 'evening'
    elif 20 <= hour < 24:
        return 'night'
    else:
        return 'unknown'
