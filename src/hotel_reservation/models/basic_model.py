"""Module with feature lookup functionalities for model registration."""

import mlflow
import numpy as np
from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

from hotel_reservation.config import ProjectConfig, Tags
from hotel_reservation.utils import configure_logging

logger = configure_logging("Hotel Reservations feature lookup")


class BasicModel:
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
        self.one_hot_encode_cols = self.config.one_hot_encode_cols
        self.features_used = self.config.features_used
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def load_data(self):
        """Load data from Databricks Delta tables."""
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_dataset")
        self.train_set = self.train_set_spark.toPandas()
        # self.train_set = self.train_set.withColumn("Booking_ID", self.train_set["Booking_ID"].cast("string"))
        # self.train_set = self.train_set.withColumn(
        #     "no_of_previous_cancellations", self.train_set["no_of_previous_cancellations"].cast("double")
        # )
        # self.train_set = self.train_set.withColumn(
        #     "no_of_previous_bookings_not_canceled",
        #     self.train_set["no_of_previous_bookings_not_canceled"].cast("double"),
        # )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_dataset").toPandas()
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[self.num_features + self.one_hot_encode_cols]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.one_hot_encode_cols]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self):
        """Perform feature engineering."""
        logger.info("🔄 Defining preprocessing pipeline...")
        # Define the log transformer using FunctionTransformer
        log_transformer = FunctionTransformer(np.log1p, validate=True)
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.config.one_hot_encode_cols),
                ("num", MinMaxScaler(), self.num_features),
                ("log", log_transformer, self.num_features),
            ],
            remainder="passthrough",
        )

        gb_model = HistGradientBoostingClassifier(
            learning_rate=self.parameters["learning_rate"], min_samples_leaf=self.parameters["min_samples_leaf"]
        )

        self.pipeline = Pipeline(steps=[("preprocessor", self.preprocessor), ("classifier", gb_model)])
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self):
        """Train the model and log results to MLflow."""
        logger.info("🚀 Starting training...")

        mlflow.set_experiment(self.experiment_name)
        logger.info(f"🚀 Starting experiment {self.experiment_name}...")

        with mlflow.start_run(tags=self.tags) as run:
            logger.info(f"Start experiment run {run.info.run_id}")
            self.run_id = run.info.run_id
            self.pipeline.fit(self.X_train, self.y_train)
            y_pred = self.pipeline.predict(self.X_test)

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

            signature = infer_signature(model_input=self.X_train, model_output=y_pred)

            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_dataset",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=self.pipeline, artifact_path="Hist-gradient-boosting-classifier-model", signature=signature
            )
        logger.info("Ended training.")

    def register_model(self):
        """Register the model with MLflow."""
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/Hist-gradient-boosting-classifier-model",
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservation_model",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservation_model",
            alias="latest-model",
            version=latest_version,
        )
        return latest_version

    def load_latest_model_and_predict(self, X):
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Args:
            X (pyspark.sql.DataFrame): The input DataFrame.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservation_model@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
