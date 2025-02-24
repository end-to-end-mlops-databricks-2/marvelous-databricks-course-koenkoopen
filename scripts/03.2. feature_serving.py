import argparse

import os
import time

import mlflow
import requests
from databricks import feature_engineering
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.serving.feature_serving import FeatureServing
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

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = f"hotel_reservation_model_fe-{args.env}"

fe = feature_engineering.FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

catalog_name = config.catalog_name
schema_name = config.schema_name
feature_table_name = f"{catalog_name}.{schema_name}.hotel_reservation_features"
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"
endpoint_name = "hotel-reservations-feature-serving"

spark.sql(f"""
          ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

# Initialize feature store manager
feature_serving = FeatureServing(
    feature_table_name=feature_table_name, feature_spec_name=feature_spec_name, endpoint_name=endpoint_name
)

# Create online table
feature_serving.create_online_table()

# Create feature spec
feature_serving.create_feature_spec()

# Deploy feature serving endpoint
feature_serving.deploy_or_update_serving_endpoint()

os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

start_time = time.time()
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_split": {"columns": ["Booking_ID"], "data": [["INN00015"]]}},
)
end_time = time.time()
execution_time = end_time - start_time

logger.info(f"Response status: {response.status_code}")
logger.info(f"Reponse text: {response.text}")
logger.info(f"Execution time: {execution_time} seconds")
