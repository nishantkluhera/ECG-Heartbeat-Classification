"""Sanity checks for project configuration (no heavy deps required)."""
from src import config


def test_feature_and_class_counts():
    assert config.N_FEATURES == 187
    assert config.N_CLASSES == 5
    assert len(config.CLASS_NAMES) == config.N_CLASSES
    assert len(config.CLASS_MAP) == config.N_CLASSES


def test_output_dirs_exist():
    # config creates these on import.
    assert config.MODEL_SAVE_DIR
    assert config.VISUALIZATION_DIR
    import os
    assert os.path.isdir(config.MODEL_SAVE_DIR)
    assert os.path.isdir(config.VISUALIZATION_DIR)
