import pandas as pd
import numpy as np
from alibi.confidence import TrustScore
import warnings
import sys
import os
warnings.filterwarnings("ignore")
import json
sys.path.append(os.path.abspath('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine'))
from utils.helper import *

def get_data():
    failures = load_failures_data('./data/model_data/failures.csv')
    components = failures['component'].unique()
    component_data = load_all_component_data(components)
    data_splits = prepare_all_data_for_training(component_data, "target_class")[0]
    models = load_all_models(components, "xgb")
    selected_features_data = fit_and_select_features(models, data_splits)
    models = retrain_models_on_selected_features(models, selected_features_data, data_splits)

    return models, selected_features_data, components,data_splits

def detect_and_replace_outliers(arr, threshold):
    inlier_array = []
    outlier_array = []
    mean_val = np.mean(arr)
    std_dev = np.std(arr)
    upper_threshold = mean_val + threshold * std_dev

    for i, val in enumerate(arr):
        if val > upper_threshold:
            outlier_array.append({"index": i, "value": arr[i]})
        else:
            inlier_array.append({"index": i, "value": arr[i]})

    return inlier_array, outlier_array

def main():
    #classes = models['GEARBOX'].classes_; 
    normal_TrustScores = []
    extreme_TrustScores = []
    models, selected_features_data, components,data_splits = get_data()
    for component in components:
        classes = models[component].classes_; 
        globals()[f"{component}_ts"] = TrustScore(alpha=.05,
                filter_type='distance_knn',
                k_filter=10,
                leaf_size=40,
                metric='euclidean',
                dist_filter_type='point')
        globals()[f"{component}_ts"].fit(selected_features_data[component][0].to_numpy(), data_splits[component][2].to_numpy(), classes=len(classes))
        globals()[f"{component}_scores"] = globals()[f"{component}_ts"].score(selected_features_data[component][1].to_numpy(), data_splits[component][3].to_numpy())
        globals()[f"{component}_normal"], globals()[f"{component}_extremes"] = detect_and_replace_outliers(globals()[f"{component}_scores"][0].tolist(), 3)

        for elem in globals()[f"{component}_normal"]:
            TrustScores = {
                             "Trust_score" : elem["value"],
                             "index" :elem["index"],
                             "component" : component
                          }
            normal_TrustScores.append(TrustScores)
        
        
        for elem in globals()[f"{component}_extremes"]:
            TrustScores = {
                            "Trust_score" : elem["value"],
                            "index" :elem["index"],
                            "component" : component
                          }
            extreme_TrustScores.append(TrustScores)
    print("Trust Scores calculated successfully")

    with open('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine/xai/JSON/normal_TrustScores.json', 'w') as json_file:
        json.dump(normal_TrustScores, json_file, indent=4)

    with open('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine/xai/JSON/extreme_TrustScores.json', 'w') as json_file:
        json.dump(extreme_TrustScores, json_file, indent=4)

if __name__ == "__main__":
        main()

