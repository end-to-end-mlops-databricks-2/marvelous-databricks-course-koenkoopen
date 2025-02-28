"""Module with feature lookup functionalities for model registration."""

import mlflow
import numpy as np
import pandas as pd
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

from hotel_reservation.config import ProjectConfig, Tags
from hotel_reservation.utils import DropColumnsTransformer, configure_logging

logger = configure_logging("Hotel Reservations feature lookup")


class HotelReservationModelWrapper(mlflow.pyfunc.PythonModel):
    """Class for the model wrapper."""

    def __init__(self, model):
        """Initialize the HotelReservationModelWrapper class."""
        self.model = model

    def predict(self, context, model_input: pd.DataFrame | np.ndarray):
        """Make predictions using the model."""
        predictions = self.model.predict(model_input)
        return {"Prediction": predictions[0]}


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
        # self.code_paths = code_path

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
        RETURNS DOUBLE
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
        self.train_set = self.train_set.withColumn("Booking_ID", self.train_set["Booking_ID"].cast("string"))
        self.train_set = self.train_set.withColumn(
            "no_of_previous_cancellations", self.train_set["no_of_previous_cancellations"].cast("double")
        )
        self.train_set = self.train_set.withColumn(
            "no_of_previous_bookings_not_canceled",
            self.train_set["no_of_previous_bookings_not_canceled"].cast("double"),
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_dataset").toPandas()
        self.test_set["no_of_previous_cancellations"] = self.test_set["no_of_previous_cancellations"].astype("float")
        self.test_set["no_of_previous_bookings_not_canceled"] = self.test_set[
            "no_of_previous_bookings_not_canceled"
        ].astype("float")
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
            exclude_columns=["update_timestamp_utc", "Booking_ID"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["cancellation_probability"] = (
            self.test_set["no_of_previous_cancellations"] / self.test_set["no_of_previous_bookings_not_canceled"]
        )
        self.X_train = self.training_df
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self):
        """Train the model and log results to MLflow."""
        logger.info("🚀 Starting training...")

        gb_model = HistGradientBoostingClassifier(
            learning_rate=self.parameters["learning_rate"], min_samples_leaf=self.parameters["min_samples_leaf"]
        )

        # Define the log transformer using FunctionTransformer
        log_transformer = FunctionTransformer(np.log1p, validate=True)

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.config.one_hot_encode_cols),
                ("num", MinMaxScaler(), self.num_features),
                ("log", log_transformer, self.num_features),
                (
                    "drop",
                    DropColumnsTransformer(columns_to_drop=self.config.columns_to_drop),
                    self.config.columns_to_drop,
                ),
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", gb_model)])

        mlflow.set_experiment(self.experiment_name)
        # mlflow.sklearn.autolog()

        # additional_pip_deps = ["pyspark==3.5.0"]
        # for package in self.code_paths:
        #     whl_name = package.split("/")[-1]
        #     additional_pip_deps.append(f"code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

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

            signature = infer_signature(self.X_train, y_pred)

            # logger.info(f"Adding conda env with packages: {additional_pip_deps}")
            # conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            mlflow.sklearn.log_model(
                HotelReservationModelWrapper(pipeline),
                "HistGradientBoostingClassifier-model-fe",
                # conda_env=conda_env,
                # code_paths=self.code_paths,
                signature=signature,
            )

            self.fe.log_model(
                model=HotelReservationModelWrapper(pipeline),
                flavor=mlflow.sklearn,
                artifact_path="HistGradientBoostingClassifier-model-fe",
                training_set=self.training_set,
                signature=signature,
                # infer_input_example=True,
            )
        logger.info("Ended training.")

    def register_model(self):
        """Register the model with MLflow.

        Returns:
            - latest_version (str): The latest version of the registered model.
        """
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
        return latest_version

    def load_latest_model_and_predict(self, X):
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Args:
            - X (pyspark.sql.DataFrame): The input DataFrame.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservation_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions

    def model_improved(self, test_set: DataFrame):
        """Evaluate the model performance on the test set."""
        X_test = test_set.drop(self.config.target)

        predictions_latest = self.load_latest_model_and_predict(X_test).withColumnRenamed(
            "prediction", "prediction_latest"
        )

        current_model_uri = f"runs:/{self.run_id}/HistGradientBoostingClassifier-model-fe"
        predictions_current = self.fe.score_batch(model_uri=current_model_uri, df=X_test).withColumnRenamed(
            "prediction", "prediction_current"
        )

        test_set = test_set.select("Booking_ID", "booking_status")

        # Join the DataFrames on the 'id' column
        df = test_set.join(predictions_current, on="Booking_ID").join(predictions_latest, on="Booking_ID")
        # Calculate the absolute error for each model
        df_pandas = df.toPandas()

        # Calculate the Mean Absolute Error (MAE) for each model
        mse_latest = mean_squared_error(df_pandas["booking_status"], df_pandas["prediction_latest"])
        mse_current = mean_squared_error(df_pandas["booking_status"], df_pandas["prediction_current"])

        # Compare models based on MAE
        logger.info(f"MSE for Current Model: {mse_current}")
        logger.info(f"MSE for Latest Model: {mse_latest}")

        if mse_current < mse_latest:
            logger.info("Current Model performs better. Registering new model.")
            return True
        else:
            logger.info("New Model performs worse. Keeping the old model.")
            return False
