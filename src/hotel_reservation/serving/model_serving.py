"""Module for feature lookup serving."""

import os
import time

import mlflow
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

from hotel_reservation.utils import configure_logging

logger = configure_logging("Hotel Reservations Feature Serving.")


class ModelServing:
    def __init__(self, model_name: str, endpoint_name: str):
        """
        Initializes the Model Serving Manager.
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def update_online_table(self, config):
        """Triggers a Databricks pipeline update and monitors its state."""

        update_response = self.workspace.pipelines.start_update(pipeline_id=config.pipeline_id, full_refresh=False)

        while True:
            update_info = self.workspace.pipelines.get_update(
                pipeline_id=config.pipeline_id, update_id=update_response.update_id
            )
            state = update_info.update.state.value

            if state == "COMPLETED":
                logger.info("Pipeline update completed successfully.")
                break
            elif state in ["FAILED", "CANCELED"]:
                logger.error("Pipeline update failed.")
                raise SystemError("Online table failed to update.")
            elif state == "WAITING_FOR_RESOURCES":
                logger.warning("Pipeline is waiting for resources.")
            else:
                logger.info(f"Pipeline is in {state} state.")

            time.sleep(30)

    def get_latest_model_version(self):
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True
    ):
        """
        Deploys the model serving endpoint in Databricks.
        :param version: str. Version of the model to deploy
        :param workload_seze: str. Workload size (number of concurrent requests). Default is Small = 4 concurrent requests
        :param scale_to_zero: bool. If True, endpoint scales to 0 when unused.
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        if version == "latest":
            entity_version = self.get_latest_model_version()
        else:
            entity_version = version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)

    def call_endpoint(self, record: list, columns: list):
        """Calls the model serving endpoint with a given input record.

        Args:
            - records (list): A list of dictionaries with records to send to the endpoint.

        Returns:
            - A dictionary of predictions.
        """
        serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{self.endpoint_name}/invocations"
        logger.info(f"Calling endpoint {serving_endpoint}")
        # response = requests.post(
        #     serving_endpoint,
        #     headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        #     json={"dataframe_records": record},
        # )

        response = requests.post(
            f"{serving_endpoint}",
            headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
            json={"dataframe_split": {"columns": columns, "data": record}},
        )

        return response.status_code, response.text
