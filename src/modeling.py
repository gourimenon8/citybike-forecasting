"""
Reusable modeling functions for training and inference
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error
from typing import Dict, Tuple

def make_lags(df: pd.DataFrame, n_lags: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.concat({
        f"{col}_lag{lag}": df[col].shift(lag).astype("float32")
        for col in df.columns
        for lag in range(1, n_lags + 1)
    }, axis=1)
    y = df.copy()
    data = pd.concat([X, y], axis=1).dropna()
    return data[X.columns], data[y.columns]

def train_baseline(y_train: pd.DataFrame, y_test: pd.DataFrame, clean_fn) -> float:
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

def train_full_lgbm(X_train, X_test, y_train, y_test, clean_fn, num_rounds=200) -> Dict[str, lgb.Booster]:
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
        first = y_train.columns[0]
        sig = infer_signature(X_train, y_train[first])
        mlflow.sklearn.log_model(models[first], "model", input_example=X_train.iloc[:1], signature=sig)
    return models

def train_top10_lgbm(project, models_full, X_train, X_test, y_train, y_test, clean_fn, num_rounds=200):
    first_station = list(models_full.keys())[0]
    imp = pd.Series(models_full[first_station].feature_importance(), index=X_train.columns)
    top10 = imp.nlargest(10).index.tolist()

    models = {}
    model_registry = project.get_model_registry()

    with mlflow.start_run(run_name="lgbm_top10_lags") as run:
        mlflow.log_param("model_type", "lightgbm_top10")
        mlflow.log_param("n_features", len(top10))

        for st in y_train.columns:
            print(f"▶️ Training model for station: {st}")
            dtrain = lgb.Dataset(X_train[top10], label=y_train[st])
            model = lgb.train({"objective": "regression", "metric": "mae"}, dtrain, num_rounds)

            preds = model.predict(X_test[top10])
            mae = mean_absolute_error(y_test[st], preds)
            mlflow.log_metric(f"mae_{clean_fn(st)}", mae)
            models[st] = model

            model_name = f"citibike_model_{clean_fn(st)}"
            model_dir = f"./{model_name}"

            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            os.makedirs(model_dir, exist_ok=True)
            input_example = X_train[top10].iloc[:1]
            sig = infer_signature(X_train[top10], y_train[st])

            mlflow.sklearn.save_model(model, path=model_dir, input_example=input_example, signature=sig)

            # Save top10 features BEFORE saving to Hopsworks
            with open(os.path.join(model_dir, "top10_features.json"), "w") as f:
                json.dump(top10, f)

            hops_model = model_registry.python.create_model(
                name=model_name,
                metrics={"mae": mae},
                description=f"LightGBM Top-10 model for {st}",
                input_example=input_example
            )

            hops_model.save(model_path=model_dir)
            print(f"✅ Registered model in Hopsworks: {model_name}")

    return models
