"""Module to define and load project configuration."""

from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Project configuration class."""

    num_features: List[str]
    cat_features: List[str]
    one_hot_encode_cols: List[str]
    features_used: List[str]
    columns_to_drop: List[str]
    target: str
    catalog_name: str
    schema_name: str
    pipeline_id: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    experiment_name_fe: Optional[str]

    @classmethod
    def from_yaml(cls, config_path: str, env: str = None):
        """Load configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML file.

        Returns:
            ProjectConfig: An instance of the ProjectConfig class.
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

        if env is not None:
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]
            config_dict["pipeline_id"] = config_dict[env]["pipeline_id"]
        else:
            config_dict["catalog_name"] = config_dict["catalog_name"]
            config_dict["schema_name"] = config_dict["schema_name"]
            config_dict["pipeline_id"] = config_dict["pipeline_id"]

        return cls(**config_dict)


class Tags(BaseModel):
    git_sha: str
    branch: str
