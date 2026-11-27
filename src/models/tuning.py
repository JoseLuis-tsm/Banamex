"""Búsqueda de hiperparámetros de XGBoost con Optuna (configuración común, CLAUDE.md §7).

Optuna con 50 pruebas, muestreador TPE con semilla 42, parada temprana de 50 rondas sobre
validación y árboles poco profundos (profundidad 2 a 4: hay pocas vintages). XGBoost y
Optuna no tienen semilla global: la semilla se pasa como parámetro a los dos.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

N_TRIALS = 50
EARLY_STOPPING = 50
MAX_ROUNDS = 3000
MAX_DEPTH_RANGE = (2, 4)


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Espacio de búsqueda común (árboles poco profundos)."""
    return {
        "max_depth": trial.suggest_int("max_depth", *MAX_DEPTH_RANGE),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 1000.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
    }


def train_booster(params: Dict[str, Any], dtrain: xgb.DMatrix, dval: xgb.DMatrix = None, num_rounds: int = MAX_ROUNDS) -> xgb.Booster:
    """Entrena un booster; con dval usa parada temprana de EARLY_STOPPING rondas."""
    kwargs = {"evals": [(dval, "validacion")], "early_stopping_rounds": EARLY_STOPPING} if dval is not None else {}
    return xgb.train(params, dtrain, num_boost_round=num_rounds, verbose_eval=False, **kwargs)


def tune(
    base_params: Dict[str, Any],
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    score_fn: Callable[[xgb.Booster], float],
    seed: int,
    n_trials: int = N_TRIALS,
) -> Tuple[Dict[str, Any], int, pd.DataFrame]:
    """Busca hiperparámetros que minimizan score_fn en validación.

    Parámetros:
        base_params: parámetros fijos (objetivo, métrica de parada, semilla, restricciones…).
        dtrain, dval: matrices de entrenamiento y validación (con etiqueta y peso).
        score_fn: recibe el booster ya detenido y devuelve la métrica a minimizar en validación.
        seed: semilla del muestreador de Optuna.

    Devuelve:
        (mejores parámetros completos, número de rondas = mejor iteración + 1, tabla de pruebas).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        booster = train_booster({**base_params, **suggest_params(trial)}, dtrain, dval)
        trial.set_user_attr("rondas", booster.best_iteration + 1)
        return score_fn(booster)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    best = study.best_trial
    return {**base_params, **best.params}, int(best.user_attrs["rondas"]), trials


def predict_best(booster: xgb.Booster, dmatrix: xgb.DMatrix) -> np.ndarray:
    """Predicción usando solo hasta la mejor iteración (si hubo parada temprana)."""
    best = getattr(booster, "best_iteration", None)
    if best is None:
        return booster.predict(dmatrix)
    return booster.predict(dmatrix, iteration_range=(0, best + 1))
