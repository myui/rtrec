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
