"""This file contains unit tests with pytest for src/config.py."""

from hotel_reservation.config import ProjectConfig


def test_project_config():
    """Unit test for the ProjectConfig class."""
    config_path = "project_config.yml"
    config = ProjectConfig.from_yaml(config_path)
    assert config.catalog_name == "koen_dev"
    assert config.schema_name == "gold_hotel_reservations"
    assert config.target == "booking_status"
    assert isinstance(config.parameters, dict)
    assert isinstance(config.num_features, list)
    assert isinstance(config.cat_features, list)
    assert isinstance(config.one_hot_encode_cols, list)
    assert isinstance(config.columns_to_drop, list)
