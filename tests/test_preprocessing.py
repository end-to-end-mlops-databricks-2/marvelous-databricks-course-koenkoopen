"""This file contains unit tests with pytest for src/preprocessing.py."""

import pandas as pd

from src.preprocessing import DataProcessor


def test_compute_quarters(spark_df, config):
    """Unit test for compute_quarters function.

    Args:
        spark_df (spark.DataFrame): The input DataFrame.
        config (ProjectConfig): The configuration object.
    """
    data_processor = DataProcessor(spark_df, config)
    data_processor.compute_quarters()
    assert data_processor.df["quarter"].tolist() == [
        "Q1",
        "Q1",
        "Q1",
        "Q2",
        "Q2",
        "Q2",
        "Q3",
        "Q3",
        "Q3",
        "Q4",
        "Q4",
        "Q4",
    ]


def test_one_hot_encode(spark_df, config):
    """Unit test for one_hot_encode function.

    Args:
        spark_df (spark.DataFrame): The input DataFrame.
        config (ProjectConfig): The configuration object.
    """
    data_processor = DataProcessor(spark_df, config)
    data_processor.one_hot_encode()
    assert data_processor.df.columns.tolist() == [
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "lead_time",
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "required_car_parking_space",
        "repeated_guest",
        "arrival_year_Q1",
    ]


def test_label_encode(spark_df, config):
    """Unit test for label_encode function.

    Args:
        spark_df (spark.DataFrame): The input DataFrame.
        config
    """
    data_processor = DataProcessor(spark_df, config)
    data_processor.label_encode()
    assert isinstance(data_processor.df["booking_status"][0], int)


def test_data_processor(config, pandas_df):
    data_processor = DataProcessor(pandas_df, config)
    data_processor.preprocess()

    assert isinstance(data_processor.df, pd.DataFrame)
    assert data_processor.df.shape == (3, 12)
    assert data_processor.df.columns.tolist() == [
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "lead_time",
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "required_car_parking_space",
        "repeated_guest",
    ]
    assert data_processor.df.isna().sum().sum() == 0, "DataFrame contains NA values"
