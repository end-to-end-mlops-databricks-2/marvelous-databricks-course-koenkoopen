import mlflow
from pyspark.sql import SparkSession

from hotel_reservation.config import ProjectConfig, Tags
from hotel_reservation.models.feature_lookup_model import FeatureLookUpModel
from hotel_reservation.utils import configure_logging

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
logger = configure_logging("Hotel Reservations Model Training")


config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Configuration loaded")
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "4ce0950880b6fdade547501027c83efd6bc5ed86", "branch": "feature/week3_serving_endpoint"}
tags = Tags(**tags_dict)

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
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

# Register the model
fe_model.register_model()
