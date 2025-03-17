import argparse

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig, Tags
from hotel_reservation.models.basic_model import BasicModel
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
basic_model = BasicModel(
    config=config,
    tags=tags,
    spark=spark,
)
# Load data
basic_model.load_data()

# Create feature table
basic_model.prepare_features()

# Train the model
basic_model.train()

# Register the model
basic_model.register_model()

latest_version = basic_model.register_model()

dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
dbutils.jobs.taskValues.set(key="model_updated", value=1)

logger.info("Model training and registration completed.")
