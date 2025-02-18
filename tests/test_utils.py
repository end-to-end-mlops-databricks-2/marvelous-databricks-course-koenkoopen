"""This file contains unit tests with pytest for src/utils.py."""

import logging

import numpy as np

from hotel_reservation.utils import configure_logging


def test_configure_logging():
    """Unit test for the function configure_logging."""
    logger = configure_logging("unit_test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.getEffectiveLevel() == logging.INFO

    logger.setLevel(logging.WARNING)
    assert logger.getEffectiveLevel() == logging.WARNING

    logger.setLevel(logging.ERROR)
    assert logger.getEffectiveLevel() == logging.ERROR
