"""Module with feature lookup functionalities for model registration."""

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hotel_reservation.config import ProjectConfig, Tags
from hotel_reservation.utils import configure_logging

logger = configure_logging("Hotel Reservations feature lookup")


class FeatureLookUpModel:
    """Class for feature lookup."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        """Initialize the FeatureLookUpModel class."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.features_used = self.config.features_used
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_reservation_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_cancellation_probability"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self):
        """Create the feature table."""
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Booking_ID STRING NOT NULL, no_of_adults DOUBLE, no_of_children DOUBLE, avg_price_per_room DOUBLE);
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT hotel_pk PRIMARY KEY(Booking_ID);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Booking_ID, no_of_adults, no_of_children, avg_price_per_room FROM {self.catalog_name}.{self.schema_name}.train_dataset"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT Booking_ID, no_of_adults, no_of_children, avg_price_per_room FROM {self.catalog_name}.{self.schema_name}.test_dataset"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self):
        """Define a function to calculate cancellation_probability"""
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(no_of_previous_cancellations DOUBLE, no_of_previous_bookings_not_canceled DOUBLE)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if no_of_previous_bookings_not_canceled == 0:
            cancellation_probability = 0 if no_of_previous_cancellations == 0 else 1
        else:
            cancellation_probability = no_of_previous_cancellations / no_of_previous_bookings_not_canceled
        return cancellation_probability
        $$
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self):
        """Load data from Databricks Delta tables."""
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_dataset").drop(
            "no_of_adults", "no_of_children", "avg_price_per_room"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_dataset").toPandas()

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self):
        """Perform feature engineering by linking data with feature tables."""
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["no_of_adults", "no_of_children", "avg_price_per_room"],
                    lookup_key="Booking_ID",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="cancellation_probability",
                    input_bindings={
                        "no_of_previous_cancellations": "no_of_previous_cancellations",
                        "no_of_previous_bookings_not_canceled": "no_of_previous_bookings_not_canceled",
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["cancellation_probability"] = (
            self.test_set["no_of_previous_cancellations"] / self.test_set["no_of_previous_bookings_not_canceled"]
        )

        self.X_train = self.training_df[self.features_used + ["cancellation_probability"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.features_used + ["cancellation_probability"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self):
        """Train the model and log results to MLflow."""
        logger.info("🚀 Starting training...")

        rf_model = HistGradientBoostingClassifier(
            learning_rate=self.parameters["learning_rate"], min_samples_leaf=self.parameters["min_samples_leaf"]
        )

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            rf_model.fit(self.X_train, self.y_train)
            y_pred = rf_model.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")

            mlflow.log_param("model_type", "HistGradientBoostingClassifier")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            signature = infer_signature(self.X_test, y_pred)

            mlflow.sklearn.log_model(rf_model, "HistGradientBoostingClassifier-model-fe", signature=signature)

            self.fe.log_model(
                model=rf_model,
                flavor=mlflow.sklearn,
                artifact_path="HistGradientBoostingClassifier-model-fe",
                training_set=self.training_set,
                infer_input_example=True,
            )
        logger.info("Ended training.")

    def register_model(self):
        """Register the model with MLflow."""
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/HistGradientBoostingClassifier-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservation_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservation_model_fe",
            alias="latest-model",
            version=latest_version,
        )

    def load_latest_model_and_predict(self, X):
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Args:
            X (pyspark.sql.DataFrame): The input DataFrame.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservation_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
