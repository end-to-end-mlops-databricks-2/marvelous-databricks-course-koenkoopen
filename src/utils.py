"""Module with utils functions."""

import logging
import os
import sys

import numpy as np
import pandas as pd


def configure_logging(name, log_file_path=None):
    """Configures the logging module for a given module.

    Returns:
        logger: The configured logger.
    """
    # Create or retrieve the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent log messages from being propagated to ancestor loggers
    logger.propagate = False

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler (to sys.stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file_path is not None:
        # Check if the directory exists; if not, create it
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def log_transform(df: pd.DataFrame, col_names: list) -> pd.DataFrame:
    """Log transform column col_name in Pandas DataFrame df.

    Args:
        df (pd.DataFrame): The Pandas DataFrame to transform.
        col_name (list): A list of strings, each representing the name of the column to transform.

    Returns:
        df: A new Pandas DataFrame with the transformed columns.
    """
    df_copy = df.copy()
    for col in col_names:
        df_copy[col] = np.log1p(df_copy[col])
    return df_copy
