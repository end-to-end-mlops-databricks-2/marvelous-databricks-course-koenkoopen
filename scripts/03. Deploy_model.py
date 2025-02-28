import argparse

from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.serving.fe_model_serving import FeatureLookupServing
from hotel_reservation.utils import configure_logging

logger = configure_logging("Hotel Reservations Deploy Model")

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

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Configuration loaded")
spark = SparkSession.builder.getOrCreate()

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = f"hotel_reservation_endpoint-{args.env}"

# Initialize feature store manager
feature_serving_manager = FeatureLookupServing(
    model_name=f"{config.catalog_name}.{config.schema_name}.hotel_reservation_model",
    endpoint_name="hotel_reservation_endpoint",
    feature_table_name=f"{config.catalog_name}.{config.schema_name}.hotel_reservation_features",
)

# Create online table
feature_serving_manager.create_online_table()

# feature_serving_manager.update_online_table(config=config)
# logger.info("Updated online table")

# Deploy feature serving endpoint
logger.info(f"Deploying model serving endpoint with version: {model_version}")
feature_serving_manager.deploy_or_update_serving_endpoint(version=model_version)
logger.info("Created model serving endpoint.")
