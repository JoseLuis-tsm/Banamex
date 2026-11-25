"""Pruebas de src/utils: lector de configuración y set_seed()."""

import random

import numpy as np

from src.utils import PROJECT_ROOT, get_path, load_config, set_seed


def test_load_config_valores_clave():
    config = load_config()
    assert config["seed"] == 42
    assert config["horizon_mob"] == 6
    assert config["split"]["train"] == {"start": 202401, "end": 202508}
    assert config["split"]["validation"] == {"start": 202509, "end": 202511}
    assert config["split"]["test"] == {"start": 202512, "end": 202603}


def test_get_path_es_absoluta_y_dentro_del_repo():
    config = load_config()
    ruta = get_path(config, "models_dir")
    assert ruta.is_absolute()
    assert ruta == PROJECT_ROOT / "models"


def test_set_seed_reproduce_numeros():
    set_seed(42)
    a = (random.random(), np.random.rand(3))
    set_seed(42)
    b = (random.random(), np.random.rand(3))
    assert a[0] == b[0]
    assert np.array_equal(a[1], b[1])


def test_set_seed_devuelve_semilla():
    assert set_seed(7) == 7
