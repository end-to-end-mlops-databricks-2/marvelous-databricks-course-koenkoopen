import argparse

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig, Tags
from hotel_reservation.models.feature_lookup_model import FeatureLookUpModel
from hotel_reservation.utils import configure_logging

logger = configure_logging("Hotel Reservations feature lookup model.")

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

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

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
fe_model = FeatureLookUpModel(
    config=config,
    tags=tags,
    spark=spark,
    # code_path=[f"{root_path}/artifacts/.internal/hotel_reservation-0.1.0-py3-none-any.whl"],
)
# Create feature table
fe_model.create_feature_table()

# Define house age feature function
fe_model.define_feature_function()

# Load data
fe_model.load_data()

# Perform feature engineering
fe_model.feature_engineering()

# Train the model
fe_model.train()

latest_version = fe_model.register_model()

dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
dbutils.jobs.taskValues.set(key="model_updated", value=1)

# Evaluate model
# Load test set from Delta table
# spark = SparkSession.builder.getOrCreate()
# test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_dataset").limit(100)
# # Drop feature lookup columns and target
# test_set = test_set.drop("no_of_adults", "no_of_children", "avg_price_per_room")
# test_set = test_set.withColumn("no_of_previous_cancellations", test_set["no_of_previous_cancellations"].cast("double"))
# test_set = test_set.withColumn("no_of_previous_bookings_not_canceled", test_set["no_of_previous_bookings_not_canceled"].cast("double"))

# model_improved = fe_model.model_improved(test_set=test_set)
# logger.info("Model evaluation completed, model improved: ", model_improved)

# if model_improved:
#     # Register the model
#     latest_version = fe_model.register_model()
#     logger.info("New model registered with version:", latest_version)
#     dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
#     dbutils.jobs.taskValues.set(key="model_updated", value=1)

# else:
#     dbutils.jobs.taskValues.set(key="model_updated", value=0)
