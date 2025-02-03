"""Module containing pytest fixtures."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from src.config import ProjectConfig


@pytest.fixture
def config():
    config = ProjectConfig.from_yaml("project_config.yml")
    yield config


@pytest.fixture
def pandas_df():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    yield df


@pytest.fixture
def spark_df():
    # Initialize a SparkSession
    spark = SparkSession.builder.master("local[1]").appName("pytest-spark").getOrCreate()

    # Create a pandas DataFrame
    pd_df = pd.DataFrame({"arrival_month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})

    # Convert the pandas DataFrame to a Spark DataFrame
    df = spark.createDataFrame(pd_df)

    yield df
