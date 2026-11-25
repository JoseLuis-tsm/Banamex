"""Fijación de la semilla para que los resultados sean reproducibles."""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> int:
    """Fija la semilla de Python, de NumPy y del hash de Python.

    XGBoost y Optuna no tienen semilla global: hay que pasarles el mismo
    valor como parámetro (random_state / seed / sampler) al crearlos.

    Parámetros:
        seed: semilla a usar (el proyecto usa 42, ver configs/config.json).

    Devuelve:
        La semilla usada, para poder pasarla a XGBoost y Optuna.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
