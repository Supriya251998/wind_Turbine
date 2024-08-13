import shap
import warnings
import sys
import os
import json
warnings.filterwarnings("ignore")
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

    return models, selected_features_data, components

def main():
    models, selected_features_data, components = get_data()
    all_components_shap_values = {}
    for component in components:
        globals()[f"{component}_explainer"] = shap.TreeExplainer(models[component])
        globals()[f"{component}_shap_values"] = globals()[f"{component}_explainer"].shap_values(selected_features_data[component][1])

        num_rows = len(selected_features_data[component][1])  # Assuming this is 729
        num_features = len(selected_features_data[component][2])  # Assuming this is 17

        component_shap_values = {}
        for i in range(num_features):
            feature_name = selected_features_data[component][2][i]
            shap_values_sum = float(globals()[f"{component}_shap_values"][:, i].sum())  # Adjust the array variable
        
        # Store feature name and sum of SHAP values in the component dictionary
            component_shap_values[feature_name] = shap_values_sum
    
        # Store component's SHAP values dictionary in the main dictionary
            all_components_shap_values[component] = component_shap_values
    print("SHAP values calculated successfully")
    json_data = json.dumps(all_components_shap_values, indent=4)
    with open('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine/xai/JSON/shap_values.json', 'w') as f:
        f.write(json_data)


if __name__ == "__main__":
            main()

