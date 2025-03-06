from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig
from hotel_reservation.serving.fe_model_serving import FeatureLookupServing
from hotel_reservation.utils import configure_logging

logger = configure_logging("Hotel Reservations Deploy Model")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Configuration loaded")
spark = SparkSession.builder.getOrCreate()

# Initialize feature store manager
feature_serving_manager = FeatureLookupServing(
    model_name=f"{config.catalog_name}.{config.schema_name}.hotel_reservation_model",
    endpoint_name="hotel_reservation_endpoint",
    feature_table_name=f"{config.catalog_name}.{config.schema_name}.hotel_reservation_features",
)

# Create online table
feature_serving_manager.create_online_table()

# Deploy feature serving endpoint
feature_serving_manager.deploy_or_update_serving_endpoint()
logger.info("Created feature serving endpoint.")

# COMMAND ----------
# Create a sample request body
required_columns = config.features_used + ["Booking_ID"]

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_dataset").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]
