import argparse

from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.preprocessing import DataProcessor
from hotel_reservation.utils import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

# Load configuration
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger = configure_logging("Hotel Reservations Preprocessing")

logger.info("Configuration loaded")

# Initialize Spark session
spark = SparkSession.builder.appName("HotelReservations").getOrCreate()

# Set the catalog and schema context
spark.sql("USE CATALOG koen_dev")
spark.sql("USE gold_hotel_reservations")

# Read data using SQL query
df = spark.sql("SELECT * FROM hotel_reservations")

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)
data_processor.preprocess()

df_train, df_test = data_processor.split_data()

data_processor.save_to_catalog(df_train, spark, table_name="train_dataset")
data_processor.save_to_catalog(df_test, spark, table_name="test_dataset")
