"""Module containing pytest fixtures."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from src.config import ProjectConfig
from src.preprocessing import DataProcessor


@pytest.fixture
def config():
    """Project configuration fixture."""
    config = ProjectConfig.from_yaml("project_config.yml")
    yield config


@pytest.fixture
def pandas_df():
    """Pandas DataFrame fixture."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    yield df


@pytest.fixture
def data_processor(config):
    """DataProcessor fixture.
    
    Args:
        config: Project configuration fixture.
    """
    # Initialize a SparkSession
    spark = SparkSession.builder.master("local[1]").appName("pytest-spark").getOrCreate()

    # Create a pandas DataFrame
    pd_df = pd.DataFrame({
        "arrival_month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "no_of_adults": [1, 2, 3, 1, 3, 4, 2, 2, 3, 5, 6, 4],
        "type_of_meal_plan": ["A", "B", "D", "A", "C", "B", "B", "D", "A", "C", "D", "A"],
        "market_segment_type": ["Online", "Offline", "Online", "Offline", "Online", "Online", "Offline", "Online", "Offline", "Online", "Offline", "Online"],
        "room_type_reserved": ["1", "2", "1", "1", "1", "2", "1", "2", "1", "2", "1", "1"],
        "booking_status": ["Canceled", "Not cancelled", "Not cancelled", "Not cancelled", "Not cancelled", "Cancelled", "Not cancelled", "Not cancelled", "Not cancelled", "Not cancelled", "Cancelled", "Not cancelled"],
        "lead_time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "avg_price_per_room": [100, 200, 150, 120, 130, 140, 110, 100, 110, 120, 130, 140],
        "no_of_children": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "no_of_weekend_nights": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "no_of_week_nights": [1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2],
        "required_car_parking_space": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "arrival_year": [2015, 2015, 2015, 2016, 2016, 2016, 2017, 2017, 2017, 2017, 2018, 2018],
        "arrival_date": [1, 10, 2, 3, 4, 7, 1, 2, 3, 4, 7, 1],
        "repeated_guest": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "no_of_previous_cancellations": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "no_of_previous_bookings_not_canceled": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "no_of_special_requests": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "Booking_ID": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        })

    # Convert the pandas DataFrame to a Spark DataFrame
    df = spark.createDataFrame(pd_df)

    data_processor = DataProcessor(df, config)
    yield data_processor
