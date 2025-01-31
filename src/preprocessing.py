"""Module with data preprocessing functions."""

from typing import Tuple

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.config import ProjectConfig
from src.utils import log_transform


class DataProcessor:
    """Class for data preprocessing."""

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        """Initialize the DataProcessor class.

        Args:
            pandas_df (pd.DataFrame): The input DataFrame.
            config (ProjectConfig): The configuration object.
        """
        self.df = pandas_df.toPandas()  # Store the DataFrame as self.df
        self.config = config  # Store the configuration

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""
        # Handle missing values and convert data types as needed

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Extract target and relevant features
        target = self.config.target
        relevant_columns = cat_features + num_features + [target]

        self.df = self.df[relevant_columns]

        self.compute_quarters("arrival_month")
        self.df = log_transform(self.df, num_features)
        self.one_hot_encode()
        self.scale_numeric_features()
        self.label_encode()
        self.df = self.df.drop(self.config.columns_to_drop,axis=1)

    def compute_quarters(self, month_column: str="arrival_month"):
        """Compute the quarter column based on the month column.
        
            Args:
                month_column (str): The name of the month column.      
        """
        self.df["quarter"] = self.df[month_column].apply(lambda x: f"Q{x // 3 + 1}")

    def one_hot_encode(self):
        """One hot encodes the categorical features."""
        self.df = pd.get_dummies(self.df, columns=self.config.one_hot_encode_cols, drop_first=True)

    def label_encode(self):
        """Label encode the target variable."""
        target_column = self.config.target
        encoder = LabelEncoder()
        self.df[target_column] = encoder.fit_transform(self.df[target_column])

    def scale_numeric_features(self):
        """Scale the numeric features using the MinMaxScaler."""
        num_features = self.config.num_features
        scaler = MinMaxScaler()
        self.df[num_features] = scaler.fit_transform(self.df[num_features])

    def split_data(self, test_size=0.2, random_state=42) -> Tuple[np.ndarray, np.ndarray]:
        """Split the DataFrame (self.df) into training and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test set.
            random_state (int): The seed used by the random number generator.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set
    
    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables.

        Args:
            train_set (pd.DataFrame): The training set.
            test_set (pd.DataFrame): The test set.
            spark (SparkSession): The Spark session.
        """

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
