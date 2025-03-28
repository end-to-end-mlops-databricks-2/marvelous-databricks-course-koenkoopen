"""Module with data preprocessing functions."""

import time
from typing import Tuple

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from hotel_reservation.config import ProjectConfig
from hotel_reservation.utils import configure_logging

logger = configure_logging("Hotel Reservations")


class DataProcessor:
    """Class for data preprocessing."""

    def __init__(self, spark_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        """Initialize the DataProcessor class.

        Args:
            spark_df (pd.DataFrame): The input DataFrame.
            config (ProjectConfig): The configuration object.
        """
        self.df = spark_df.toPandas()  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark
        logger.info("DataProcessor initialized")

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""
        logger.info("Preprocessing data...")

        try:
            # Handle numeric features
            num_features = self.config.num_features
            for col in num_features:
                if col not in self.df.columns:
                    logger.error(f"Column {col} not found in the DataFrame, skipping conversion.")
                else:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        except Exception as e:
            logger.error(f"Unexpected error while converting numerical features: {e}")
            raise Exception(f"Unexpected error while converting numerical features: {e}") from e

        # Handle missing values and convert data types as needed
        self.df.fillna(
            {
                "no_of_adults": self.df["no_of_adults"].mean(),
                "no_of_children": "None",
                "no_of_weekend_nights": 0,
                "no_of_week_nights": 0,
                "required_car_parking_space": 0,
                "lead_time": self.df["lead_time"].mean(),
                "arrival_year": "None",
                "arrival_month": "None",
                "arrival_date": "None",
                "repeated_guest": 0,
                "no_of_previous_cancellations": 0,
                "no_of_previous_bookings_not_canceled": 0,
                "avg_price_per_room": self.df["avg_price_per_room"].mean(),
                "no_of_special_requests": 0,
                "type_of_meal_plan": "None",
                "room_type_reserved": "None",
                "market_segment_type": "None",
            },
            inplace=True,
        )

        try:
            # Convert categorical features to the appropriate type
            cat_features = self.config.cat_features
            for cat_col in cat_features:
                if cat_col not in self.df.columns:
                    logger.warning(f"Column {cat_col} not found in the DataFrame, skipping conversion.")
                else:
                    self.df[cat_col] = self.df[cat_col].astype("category")
        except Exception as e:
            logger.error(f"Unexpected error while converting categorical features: {e}")
            raise Exception(f"Unexpected error while converting categorical features: {e}") from e

        # Extract target and relevant features
        target = self.config.target
        relevant_columns = cat_features + num_features + [target] + ["Booking_ID"]
        self.df = self.df[relevant_columns]
        self.df["Booking_ID"] = self.df["Booking_ID"].astype("str")

        self.df = self.df[relevant_columns]
        self.compute_quarters("arrival_month")
        self.label_encode()
        self.df.columns = self.df.columns.str.replace(" ", "_")

    def compute_quarters(self, month_column: str = "arrival_month"):
        """Compute the quarter column based on the month column.

        Args:
            month_column (str): The name of the month column.
        """
        try:
            logger.info(f"Computing quarters based on {month_column}...")
            self.df["quarter"] = self.df[month_column].apply(lambda x: f"Q{(x - 1) // 3 + 1}")
        except KeyError as e:
            logger.error(f"While computing quarters, the column {month_column} does not exist in the DataFrame: {e}")
            raise KeyError(
                f"While computing quarters, the column {month_column} does not exist in the DataFrame: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error occurred while computing quarters: {e}")
            raise Exception(f"Unexpected error occurred while computing quarters: {e}") from e

    def label_encode(self):
        """Label encode the target variable."""
        logger.info("Label encoding target variable...")
        try:
            target_column = self.config.target
            encoder = LabelEncoder()
            self.df[target_column] = encoder.fit_transform(self.df[target_column])
        except KeyError as e:
            logger.error(f"Target column {target_column} does not exist in the DataFrame: {e}")
            raise KeyError(f"Target column {target_column} does not exist in the DataFrame: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error occurred while label encoding: {e}")
            raise Exception(f"Unexpected error occurred while label encoding: {e}") from e

    def split_data(self, test_size=0.2, random_state=42) -> Tuple[np.ndarray, np.ndarray]:
        """Split the DataFrame (self.df) into training and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test set.
            random_state (int): The seed used by the random number generator.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the training and test sets.
        """
        logger.info(f"Splitting data into training and test sets (test_size={test_size})...")
        try:
            train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        except Exception as e:
            logger.error(f"Unexpected error occurred while splitting data: {e}")
            raise Exception(f"Unexpected error occurred while splitting data: {e}") from e
        return train_set, test_set

    def save_to_catalog(self, df: pd.DataFrame, spark: SparkSession, table_name: str):
        """Save the train and test sets into Databricks tables.

        Args:
            df (pd.DataFrame): The dataframe to save.
            spark (SparkSession): The Spark session.
            table_name (str): The name of the table to create.
        """
        logger.info("Saving train and test sets to Databricks tables...")
        try:
            catalog_dest = f"{self.config.catalog_name}.{self.config.schema_name}.{table_name}"

            spark_df = spark.createDataFrame(df).withColumn(
                "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
            )

            spark_df.write.mode("append").saveAsTable(catalog_dest)

            spark.sql(
                f"""
                ALTER TABLE {catalog_dest}
                SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
                """
            )
        except Exception as e:
            logger.error(f"Unexpected error occurred while saving to Databricks tables: {e}")
            raise Exception(f"Unexpected error occurred while saving to Databricks tables: {e}") from e

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_dataset "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_dataset "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


def generate_synthetic_data(df, config, drift=False, num_rows=10):
    """Generates synthetic data based on the distribution of the input DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        config (ProjectConfig): The configuration object.
        drift (bool, optional): Whether to apply drift to the synthetic data. Defaults to False.
        num_rows (int, optional): The number of rows to generate. Defaults to 10.

    Returns:
        pandas.DataFrame: The generated synthetic data.
    """
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == "Booking_ID":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            if column in {"arrival_year"}:  # Handle year-based columns separately
                synthetic_data[column] = np.random.randint(df[column].min(), df[column].max() + 1, num_rows)
            else:
                synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Convert relevant numeric columns to integers
    int_columns = config.num_features
    for col in df.columns.intersection(int_columns):
        synthetic_data[col] = synthetic_data[col].astype(np.int32)

    timestamp_base = int(time.time() * 1000)
    synthetic_data["Booking_ID"] = [str(timestamp_base + i) for i in range(num_rows)]

    if drift:
        # Skew the top features to introduce drift
        top_features = ["lead_time", "avg_price_per_room"]  # Select top 2 features
        for feature in top_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 2

        # Set YearBuilt to within the last 2 years
        current_year = pd.Timestamp.now().year
        if "arrival_year" in synthetic_data.columns:
            synthetic_data["arrival_year"] = np.random.randint(current_year - 2, current_year + 1, num_rows)

    return synthetic_data
