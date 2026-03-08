"""
SpineIDS Clinical ML - Model Definitions
=========================================
Factory and utilities for the five classifiers used in Part 1:

    1. Logistic Regression  (ElasticNet regularisation)
    2. Random Forest
    3. XGBoost
    4. LightGBM
    5. CatBoost

All boosting libraries are treated as optional dependencies; the factory
raises an informative error if a requested model is unavailable.
"""

from typing import Any, Dict, Optional

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from .config import DEFAULT_PARAMS, HYPERPARAM_SPACE, MODELS_NEED_SCALING, RANDOM_SEED


class ModelFactory:
    """Factory for instantiating and querying classifier objects."""

    MODEL_CLASSES: Dict[str, Any] = {
        'LogisticRegression': LogisticRegression,
        'RandomForest':       RandomForestClassifier,
    }

    if HAS_XGBOOST:
        MODEL_CLASSES['XGBoost']  = XGBClassifier
    if HAS_LIGHTGBM:
        MODEL_CLASSES['LightGBM'] = LGBMClassifier
    if HAS_CATBOOST:
        MODEL_CLASSES['CatBoost'] = CatBoostClassifier

    @classmethod
    def get_available_models(cls) -> list:
        """Return the list of currently available model names."""
        return list(cls.MODEL_CLASSES.keys())

    @classmethod
    def create_model(
        cls,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
        use_default: bool = True,
    ) -> Any:
        """
        Instantiate a classifier.

        Args:
            model_name:  Registered model name.
            params:      Parameter overrides.  When ``use_default`` is True,
                         these are merged on top of the default parameter set.
            use_default: Seed the parameter dict from ``DEFAULT_PARAMS`` before
                         applying any overrides.

        Returns:
            Unfitted classifier instance.
        """
        if model_name not in cls.MODEL_CLASSES:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(cls.MODEL_CLASSES.keys())}"
            )

        model_class = cls.MODEL_CLASSES[model_name]

        if use_default and model_name in DEFAULT_PARAMS:
            final_params = DEFAULT_PARAMS[model_name].copy()
            if params is not None:
                final_params.update(params)
        else:
            final_params = params or {}

        return model_class(**final_params)

    @classmethod
    def needs_scaling(cls, model_name: str) -> bool:
        """Return True if the model requires standardised input features."""
        return model_name in MODELS_NEED_SCALING

    @classmethod
    def get_param_space(cls, model_name: str) -> Dict[str, tuple]:
        """Return the Optuna search space definition for *model_name*."""
        if model_name not in HYPERPARAM_SPACE:
            raise ValueError(f"No hyperparameter space defined for '{model_name}'")
        return HYPERPARAM_SPACE[model_name]

    @classmethod
    def get_default_params(cls, model_name: str) -> Dict[str, Any]:
        """Return a copy of the default parameter dict for *model_name*."""
        if model_name not in DEFAULT_PARAMS:
            raise ValueError(f"No default parameters defined for '{model_name}'")
        return DEFAULT_PARAMS[model_name].copy()


def sample_params_from_space(
    param_space: Dict[str, tuple],
    trial,
) -> Dict[str, Any]:
    """
    Sample a parameter configuration from *param_space* using an Optuna trial.

    Args:
        param_space: Search space dict as defined in ``config.HYPERPARAM_SPACE``.
        trial:       ``optuna.trial.Trial`` object.

    Returns:
        Dictionary of sampled parameter values.
    """
    params: Dict[str, Any] = {}

    for name, spec in param_space.items():
        kind = spec[0]

        if kind == 'fixed':
            params[name] = spec[1]
        elif kind == 'int':
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == 'float':
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == 'float_log':
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == 'categorical':
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown parameter type '{kind}' for '{name}'")

    return params
