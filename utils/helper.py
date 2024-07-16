import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.feature_selection import SelectFromModel

# load the failures data

def load_failures_data(failures_path):
    return pd.read_csv(failures_path, sep=',')

def load_all_component_data(components):
    encoder = LabelEncoder()
    component_data = {}
    for component in components:
        df = pd.read_csv(f'../data/model_data/labelled_data_{component}.csv', sep=',')
        df['turbine_id'] = encoder.fit_transform(['turbine_id'] * df.shape[0])
        df = df.set_index('timestamp')
        component_data[component] = df
    return component_data

def prepare_all_data_for_training(component_data, class_target_name):
    data_splits = {}
    for component, df in component_data.items():
        X = df.drop(columns=['component', class_target_name])
        y = df[class_target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # return all X_train, X_test, y_train, y_test
        data_splits[component] = (X_train, X_test, y_train, y_test)
    return data_splits

def load_all_models(components, model_name):
    models = {}
    for component in components:
        with open(f'../models/selected-{component}.pickle', 'rb') as file:
            models[component] = pickle.load(file)
    return models

def fit_and_select_features(models, data_splits):
    selected_features_data = {}
    for component, model in models.items():
        X_train, X_test, y_train, y_test = data_splits[component]
        model.fit(X_train, y_train)
        params = model.get_params()
        selector = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=params.get('max_features', None))
        selected_features = X_train.columns[selector.get_support()]
        selected_features_train = X_train[selected_features]
        selected_features_train.reset_index(drop=True, inplace=True)
        selected_features_test = X_test[selected_features]
        selected_features_test.reset_index(drop=True, inplace=True)
        selected_features_data[component] = (selected_features_train, selected_features_test, selected_features)
    return selected_features_data

def retrain_models_on_selected_features(models, selected_features_data, data_splits):
    for component, model in models.items():
        selected_features_train, _, _ = selected_features_data[component]
        _, _, y_train, _ = data_splits[component]
        model.fit(selected_features_train, y_train)
    return models

