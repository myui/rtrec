import numpy as np
import pandas as pd


def convert_dtypes(df):
    def parse_numeric(x, downcast):
        try:
            return pd.to_numeric(x, errors="raise", downcast=downcast)
        except Exception:
            return x
    df.loc[:, df.dtypes == np.integer] = df.loc[:, df.dtypes == np.integer].apply(parse_numeric, downcast="integer")
    df.loc[:, df.dtypes == np.inexact] = df.loc[:, df.dtypes == np.inexact].apply(parse_numeric, downcast="float")
    return df.convert_dtypes()
