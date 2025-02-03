"""This file contains unit tests with pytest for src/utils.py."""

import logging

import numpy as np

from src.utils import configure_logging, log_transform


def test_configure_logging():
    """Unit test for the function configure_logging."""
    logger = configure_logging("unit_test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.getEffectiveLevel() == logging.INFO

    logger.setLevel(logging.WARNING)
    assert logger.getEffectiveLevel() == logging.WARNING

    logger.setLevel(logging.ERROR)
    assert logger.getEffectiveLevel() == logging.ERROR


def test_log_transform(pandas_df):
    """Unit test for log_transform function.

    Args:
        pandas_df (pd.DataFrame): The input DataFrame.
    """
    transformed_array = log_transform(pandas_df, ["col1", "col2"])
    assert np.allclose(transformed_array, np.log1p(pandas_df[["col1", "col2"]]))
