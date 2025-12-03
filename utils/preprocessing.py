import pandas as pd

def preprocess_live_data(df):
    # Add your cleaning steps here
    df = df.dropna()
    df = df.select_dtypes(include=['float64', 'int64'])  # Keep numeric only
    return df
