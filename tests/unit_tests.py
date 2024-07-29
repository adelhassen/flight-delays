import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from main import read_data, feature_engineering

def test_read_data():

    data = read_data("data/july_2023.csv.zip")

    expected_cols = [
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "AIRLINE",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "ARR_DELAY",
        "ARR_DEL15"
    ]

    actual_cols = data.columns.tolist()

    assert actual_cols == expected_cols


def test_feature_engineering():

    data = {
        'DAY_OF_MONTH': 1,
        'DAY_OF_WEEK': 6,
        'AIRLINE': 'Endeavor Air Inc.',
        'ORIGIN': 'AUS',
        'DEST': 'RDU',
        'CRS_DEP_TIME': 1007,
        'CRS_ARR_TIME': 1412,
        'CRS_ELAPSED_TIME': 185.0,
        'DISTANCE': 1162.0,
        'ARR_DELAY': -8.0,
        'ARR_DEL15': 0.0
    }

    df = pd.DataFrame([data])
    
    actual_features, y = feature_engineering(df)
    actual_features = actual_features.drop(["DAY_OF_MONTH_SIN", "DAY_OF_MONTH_COS", "DAY_OF_WEEK_SIN", "DAY_OF_WEEK_COS"], axis=1).loc[0,:].to_dict()

    expected_features = {
        'AIRLINE': 'Endeavor Air Inc.',
        'ORIGIN': 'AUS',
        'DEST': 'RDU',
        'CRS_ELAPSED_TIME': 185.0,
        'CRS_DEP_TIME_MINUTES_SIN': 0.4733196671848435,
        'CRS_DEP_TIME_MINUTES_COS': -0.8808907382053855,
        'CRS_ARR_TIME_MINUTES_SIN': -0.5446390350150271,
        'CRS_ARR_TIME_MINUTES_COS': -0.838670567945424
    }

    assert actual_features == expected_features

if __name__ == "__main__":
    test_read_data()
    test_feature_engineering()