"""This file contains unit tests with pytest for src/preprocessing.py."""

import pandas as pd
import pytest

from hotel_reservation.preprocessing import DataProcessor


def test_compute_quarters(data_processor):
    """Unit test for compute_quarters function.

    Args:
        data_processor (DataProcessor): The DataProcessor object.
    """
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


def test_compute_quarters_error(data_processor):
    """Unit test for compute_quarters function to check on correct error raising.

    Args:
        data_processor (DataProcessor): The DataProcessor object.
    """
    with pytest.raises(KeyError):
        data_processor.compute_quarters(month_column="test_month")


def test_one_hot_encode(data_processor):
    """Unit test for one_hot_encode function.

    Args:
        data_processor (DataProcessor): The DataProcessor object.
    """
    data_processor.compute_quarters()
    data_processor.one_hot_encode()
    assert data_processor.df.columns.tolist() == [
        "arrival_month",
        "no_of_adults",
        "booking_status",
        "lead_time",
        "avg_price_per_room",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "required_car_parking_space",
        "arrival_year",
        "arrival_date",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "no_of_special_requests",
        "Booking_ID",
        "market_segment_type_Online",
        "type_of_meal_plan_B",
        "type_of_meal_plan_C",
        "type_of_meal_plan_D",
        "room_type_reserved_2",
        "quarter_Q2",
        "quarter_Q3",
        "quarter_Q4",
    ]


def test_one_hot_encode_keyerror(data_processor):
    """Unit test for one_hot_encode function when missing a column.

    Args:
        data_processor (DataProcessor): The DataProcessor object.
    """
    with pytest.raises(KeyError):
        data_processor.one_hot_encode()


def test_label_encode(data_processor):
    """Unit test for label_encode function.

    Args:
        data_processor (DataProcessor): The DataProcessor object.
    """
    data_processor.label_encode()
    assert data_processor.df["booking_status"][0] == 0


def test_label_encode_keyerror(data_processor):
    """Unit test for label_encode function when the target column is not in the dataframe.

    Args:
        data_processor (DataProcessor): The DataProcessor object.
    """
    data_processor.config.target = "test_target"
    with pytest.raises(KeyError):
        data_processor.label_encode()


def test_scale_numeric_features(data_processor):
    """Unit test for scale_numeric_features function.

    Args:
        data_processor (DataProcessor): The DataProcessor object.
    """
    data_processor.scale_numeric_features()
    assert data_processor.df["lead_time"].min() >= 0
    assert data_processor.df["lead_time"].max() <= 1


def test_scale_numeric_features_keyerror(data_processor):
    """Unit test for scale_numeric_features function correctly raising a keyerror.

    Args:
        data_processor (DataProcessor): The DataProcessor object.
    """
    data_processor.config.num_features = ["test_feature"]
    with pytest.raises(KeyError):
        data_processor.scale_numeric_features()


def test_split_data(data_processor):
    """Unit test for label_encode function.

    Args:
        spark_df (spark.DataFrame): The input DataFrame.
        config (ProjectConfig): The configuration object.
    """
    train_set, test_set = data_processor.split_data()
    assert train_set.shape == (9, 19)
    assert test_set.shape == (3, 19)


def test_data_processor(data_processor):
    """Unit test for the DataProcessor class.

    Args:
        data_processor (DataProcessor): The DataProcessor object.
    """
    assert isinstance(data_processor, DataProcessor)
    assert isinstance(data_processor.df, pd.DataFrame)


def test_preprocess(data_processor):
    """Unit test for the preprocess function.

    Args:
        data_processor (DataProcessor): The DataProcessor object.
    """
    data_processor.config.columns_to_drop = ["Booking_ID", "arrival_date", "arrival_year"]
    data_processor.preprocess()

    assert data_processor.df.isna().sum().sum() == 0, "DataFrame contains NA values"
