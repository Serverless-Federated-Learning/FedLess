import json
from itertools import zip_longest
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf


def resource_folder_path() -> Path:
    """
    Get path to test resource directory
    :return: Path
    """
    return Path(__file__).parent / "res"


def get_error_function(error: Exception.__class__, *arguments):
    # noinspection PyUnusedLocal
    def f(*args, **kwargs):
        raise error(*arguments)

    return f


def are_weights_equal(weight_old: List[np.ndarray], weights_new: List[np.ndarray]):
    for a, b in zip_longest(weight_old, weights_new):
        if not np.allclose(a, b):
            return False
    return True


def is_optimizer_state_preserved(
    optimizer_old: tf.keras.Model, optimizer_new: tf.keras.Model
):
    if optimizer_old is None or optimizer_new is None:
        return False
    if not optimizer_old.get_config() == optimizer_new.get_config():
        return False
    if not are_weights_equal(optimizer_old.get_weights(), optimizer_new.get_weights()):
        return False
    return True


def is_model_trainable(model: tf.keras.Model, data: Tuple[np.ndarray, np.ndarray]):
    features, labels = data
    try:
        model.fit(features, labels, batch_size=1)
    except (RuntimeError, ValueError):
        return False
    return True


def is_valid_json(json_str):
    try:
        json.loads(json_str)
    except (JSONDecodeError, ValueError):
        return False
    return True
