import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier 
from sklearn.model_selection import  cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import mlflow
import optuna
import warnings
import sys
import os
from sklearn.ensemble import ExtraTreesClassifier
sys.path.append(os.path.abspath('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine'))
from utils.helper import *
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)
mlflow.set_tracking_uri("http://0.0.0.0:80")
experiment_name = "predictive_maintenance"
experiment_id = mlflow.create_experiment(experiment_name)

def get_data():
    failures = load_failures_data('./data/model_data/failures.csv')
    components = failures['component'].unique()
    component_data = load_all_component_data(components)
    data_splits = prepare_all_data_for_training(component_data, "target_class")[0]
    return data_splits , components
data_splits = get_data()[0]

def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def objective(trial, component):
    """
    Objective function for Optuna study to optimize hyperparameters for the XGBoost classifier
    """

    # define the search space for the hyperparameters
    search_space = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
        'max_depth': trial.suggest_categorical('max_depth', [10, 20, 40, 80]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'lambda': 1.0,
        'alpha': 1.0,
        'max_features': trial.suggest_int('max_features', 5, 20)
    }
    
    # create the XGBoost classifier with the hyperparameters
    model = XGBClassifier(**search_space)
 
    # apply feature selection
  
    selector = SelectFromModel(model, threshold=-np.inf, max_features=search_space['max_features'])

    pipeline = Pipeline([
        ('feature_selection', selector),
        ('model', model)
    ])
    pipeline.fit(data_splits[component][0], data_splits[component][2])


    # evaluate the model with cross-validation
    score = f1_score(data_splits[component][3], pipeline.predict(data_splits[component][1]), average='weighted')

    return score

def main():
    mlflow.set_experiment(experiment_name)
    trials = 100
    components = get_data()[1]
    
    for component in components:
        globals()[f"{component}_study"] = optuna.create_study(direction='maximize')
        globals()[f"{component}_study"].optimize(lambda trial: objective(trial, component), n_trials=trials, callbacks=[champion_callback])
        best_params = globals()[f"{component}_study"].best_params
        best_score = globals()[f"{component}_study"].best_value
        print(f"\nBest score for {component} component: {best_score}\n")
        print(f"Best parameters for {component} component: {best_params}")

        with mlflow.start_run(experiment_id=experiment_id, run_name=component):
            mlflow.log_params(best_params)
            mlflow.log_metrics({'score': best_score})
            
            mlflow.set_tags({
                "project": "Thesis",
                "optimizer_engine": "optuna",
                "model_family": "xgboost",
                "feature_set_version": 1,
                "component": component
            })

            # Save the entire pipeline (with feature selection)
            selector = SelectFromModel(XGBClassifier(**best_params), threshold=-np.inf, max_features=best_params['max_features'])
            pipeline = Pipeline([
                ('feature_selection', selector),
                ('model', XGBClassifier(**best_params))
            ])
            
            pipeline.fit(data_splits[component][0], data_splits[component][2])

            artifact_path = "model"
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=artifact_path,
                metadata={"model_data_version": 1}
            )
            
            # Save the model locally
            globals()[f"{component}_best_model"] = pipeline
    
    return {f"{component}_best_model": globals()[f"{component}_best_model"] for component in components}

if __name__ == "__main__":
    study = main()
    if not os.path.exists('./models'):
        os.makedirs('./models')
    '''for component, model in study.items():
        with open(f'./models/{component}.pkl', 'wb') as file:
            pickle.dump(model, file)'''


# to run this file first you need to activate the Ml flow
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 80 
# After this command you can see the experiments in the mlflow server. I have to start this port in this file, only then I can access the experiments

