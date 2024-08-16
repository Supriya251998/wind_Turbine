from alibi.explainers import AnchorTabular
import time
from sklearn.model_selection import  train_test_split
import warnings
warnings.filterwarnings("ignore")
import sys
import os
from sklearn.model_selection import  cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
sys.path.append(os.path.abspath('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine'))
from utils.helper import *

def get_data():
    failures = load_failures_data('./data/model_data/failures.csv')
    components = failures['component'].unique()
    component_data = load_all_component_data(components)
    data_splits = prepare_all_data_for_training(component_data, "target_class")
    models = load_all_models(components, "xgb")
    selected_features_data = fit_and_select_features(models, data_splits)
    models = retrain_models_on_selected_features(models, selected_features_data, data_splits)

    return models, components,selected_features_data,data_splits

def train_run(X, y, X_valid, y_valid, model):
    clf = model
    clf.fit(X, y)
    y_valid_pred = clf.predict(X_valid)
    y_train_pred = clf.predict(X)
    f1_train = f1_score(y, y_train_pred, average="weighted")
    precision_train = precision_score(y, y_train_pred, average="weighted")
    recall_train = recall_score(y, y_train_pred, average="weighted")
    f1 = f1_score(y_valid, y_valid_pred, average="weighted")
    precision = precision_score(y_valid, y_valid_pred, average="weighted")
    recall = recall_score(y_valid, y_valid_pred, average="weighted")

    scores = {
        "f1_train": f1_train,
        "precision_train": precision_train,
        "recall_train": recall_train,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    return scores

def main():
    models, components, selected_features_data, data_splits = get_data()
    for component in components:
        model = models[component]
        selected_features_train, selected_features_test, selected_features = selected_features_data[component]
        X_train, X_test, y_train, y_test = data_splits[component]
        model.fit(selected_features_train, y_train)
        scores = train_run(selected_features_train, y_train, selected_features_test, y_test, model)
        print(f"Component: {component}")
        print(scores)

if __name__ == "__main__":
    main()