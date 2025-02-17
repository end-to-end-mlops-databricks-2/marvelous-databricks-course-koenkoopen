%pip install databricks-feature-engineering==0.6 databricks-sdk==0.32.0
%restart_python

import os

from hotel_reservation.config import ProjectConfig
from hotel_reservation.serving.fe_model_serving import FeatureLookupServing
from hotel_reservation.utils import configure_logging

logger = configure_logging("Hotel Reservations Deploy Model")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Configuration loaded")

# Initialize feature store manager
feature_serving_manager = FeatureLookupServing(model_name=f'{config.catalog_name}.{config.schema_name}.hotel_reservation_model_fe', endpoint_name='hotel_reservation_endpoint_koen', feature_table_name=f'{config.catalog_name}.{config.schema_name}.hotel_reservation_features')

# Create online table
# feature_serving_manager.create_online_table()

# Deploy feature serving endpoint
feature_serving_manager.deploy_or_update_serving_endpoint()
logger.info("Created feature serving endpoint.")

# COMMAND ----------
# Create a sample request body
required_columns = config.features_used + ["Booking_ID"]

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_dataset").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# Call the endpoint with one sample record
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")


status_code, response_text = feature_serving_manager.call_endpoint([sampled_records[0]])
logger.info(f"Response Status: {status_code}")
logger.info(f"Response Text: {response_text}")