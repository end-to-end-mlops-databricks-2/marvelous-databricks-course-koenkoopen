from hotel_reservation.config import ProjectConfig
from hotel_reservation.serving.fe_model_serving import FeatureLookupServing
from hotel_reservation.utils import configure_logging

logger = configure_logging("Hotel Reservations Deploy Model")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
logger.info("Configuration loaded")

# Initialize feature store manager
feature_manager = FeatureServing(config)

# Load data and model
feature_manager.load_data()
feature_manager.load_model()

# Create feature table and enable Change Data Feed
feature_manager.create_feature_table()

logger.info("Created feture table.")

# Create online table
feature_manager.create_online_table()
logger.info("Created feture online table.")

# Create feature spec
feature_manager.create_feature_spec()
logger.info("Created feture spec for deployment.")

# Deploy feature serving endpoint
feature_manager.deploy_serving_endpoint()
logger.info("Created feature serving endpoint.")