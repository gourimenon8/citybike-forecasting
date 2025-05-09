"""
Reusable modeling functions for training and inference
"""
"""
Reusable modeling functions for training and inference
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from mlflow.models import infer_signature
import mlflow
from typing import Tuple, Dict
from mlflow.tracking import MlflowClient

def make_lags(df: pd.DataFrame, n_lags: int = 28
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # X: each station × lag as a feature
    X = pd.concat({
        f"{col}_lag{lag}": df[col].shift(lag).astype("float32")
        for col in df.columns
        for lag in range(1, n_lags + 1)
    }, axis=1)
    y = df.copy()
    data = pd.concat([X, y], axis=1).dropna()
    return data[X.columns], data[y.columns]

def train_baseline(y_train: pd.DataFrame, y_test: pd.DataFrame,
                   clean_fn) -> float:
    """
    Log a baseline mean model to MLflow, return average MAE.
    """
    with mlflow.start_run(run_name="baseline_mean"):
        mlflow.log_param("model_type", "mean_baseline")
        preds = np.tile(y_train.mean().values, (len(y_test), 1))
        preds_df = pd.DataFrame(preds, index=y_test.index, columns=y_test.columns)
        for st in y_test.columns:
            m = mean_absolute_error(y_test[st], preds_df[st])
            mlflow.log_metric(f"mae_{clean_fn(st)}", m)
        avg = mean_absolute_error(y_test.values, preds_df.values)
        mlflow.log_metric("mae_avg", avg)
        return avg

def train_full_lgbm(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.DataFrame, y_test: pd.DataFrame,
                    clean_fn, num_rounds: int = 200
                   ) -> Dict[str, lgb.Booster]:
    """
    Train one LightGBM per station on all lag features.
    Returns a dict of trained models keyed by station.
    """
    models = {}
    with mlflow.start_run(run_name="lgbm_28lags_full"):
        mlflow.log_param("model_type", "lightgbm_full")
        mlflow.log_param("n_lags", int(len(X_train.columns) / len(y_train.columns)))
        for st in y_train.columns:
            dtrain = lgb.Dataset(X_train, label=y_train[st])
            model = lgb.train({"objective": "regression", "metric": "mae"}, dtrain, num_rounds)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test[st], preds)
            mlflow.log_metric(f"mae_{clean_fn(st)}", mae)
            models[st] = model
        # Log artifact for the first station as representative
        first = y_train.columns[0]
        sig = infer_signature(X_train, y_train[first])
        mlflow.sklearn.log_model(models[first], "model",
                                 input_example=X_train.iloc[:1], signature=sig)
    return models

def train_top10_lgbm(models_full: Dict[str, lgb.Booster],
                     X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.DataFrame, y_test: pd.DataFrame,
                     clean_fn, num_rounds: int = 200
                    ) -> Dict[str, lgb.Booster]:
    """
    Train LightGBM using top-10 features by importance from models_full.
    Registers the best model in MLflow Model Registry.
    """
    import pandas as pd
    # derive top10 from first station model
    first = list(models_full.keys())[0]
    imp = pd.Series(models_full[first].feature_importance(), index=X_train.columns)
    top10 = imp.nlargest(10).index.tolist()
    models = {}
    with mlflow.start_run(run_name="lgbm_top10_lags") as run:
        mlflow.log_param("model_type", "lightgbm_top10")
        mlflow.log_param("n_features", len(top10))
        for st in y_train.columns:
            dtrain = lgb.Dataset(X_train[top10], label=y_train[st])
            model = lgb.train({"objective": "regression", "metric": "mae"}, dtrain, num_rounds)
            preds = model.predict(X_test[top10])
            mae = mean_absolute_error(y_test[st], preds)
            mlflow.log_metric(f"mae_{clean_fn(st)}", mae)
            models[st] = model
        # log & register first station model
        sig = infer_signature(X_train[top10], y_train[y_train.columns[0]])
        mlflow.sklearn.log_model(models[y_train.columns[0]], "model",
                                 input_example=X_train[top10].iloc[:1], signature=sig)
        run_id = run.info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", "citibike_best_model")
    return models
