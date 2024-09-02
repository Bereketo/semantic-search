import pandas as pd

data_path = "./data/imdb_top_1000.csv"


def load_data(df):
    """Load data from csv"""
    df = pd.read_csv(data_path)
    return df


def preprocess_data(df):
    """Select relevant columns and preprocess the text"""
    df_selected = df[
        [
            "Series_Title",
            "Genre",
            "Overview",
            "Director",
            "Star1",
            "Star2",
            "Star3",
            "Star4",
        ]
    ]
    return df_selected
