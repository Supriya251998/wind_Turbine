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
    data_splits = prepare_all_data_for_training(component_data, "target_class")[0]
    models = load_all_models(components, "xgb")
    selected_features_data = fit_and_select_features(models, data_splits)
    models = retrain_models_on_selected_features(models, selected_features_data, data_splits)

    return models, selected_features_data, components

     
     

def main():
    models, selected_features_data, components = get_data()
    all_components_shap_values = {}
    all_components_shap_values_instance = {}

    for component in components:
        # Initialize the explainer and SHAP values for the current component
        explainer = shap.TreeExplainer(models[component].named_steps['model'])
        shap_values = explainer.shap_values(selected_features_data[component][1])

        num_rows = len(shap_values)  # Assuming this is 729
        num_features = len(selected_features_data[component][2])  # Assuming this is 17

        # Store feature names and their summed SHAP values
        component_shap_values = {}
        for i in range(num_features):
            feature_name = selected_features_data[component][2][i]
            shap_values_sum = float(shap_values[:, i].sum())  # Sum of SHAP values for each feature
            component_shap_values[feature_name] = shap_values_sum

        # Store the SHAP values for this component in the main dictionary
        all_components_shap_values[component] = component_shap_values

        # Store instance SHAP values
        component_shap_values_instance = {}
        for i in range(num_rows):
            component_shap_values_instance[i] = {component: {}}
            for j in range(num_features):
                feature_name = selected_features_data[component][2][j]
                shap_value = float(shap_values[i, j])
                component_shap_values_instance[i][component][feature_name] = shap_value

        # Store component's SHAP values dictionary in the instance dictionary
        all_components_shap_values_instance[component] = component_shap_values_instance

    print("SHAP values calculated successfully")
    
    # Save all SHAP values in a JSON file for components
    json_data = json.dumps(all_components_shap_values, indent=4)
    with open('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine/xai/JSON/shap_values.json', 'w') as f:
        f.write(json_data)

    # Save all instance SHAP values in a separate JSON file
    json_data_instance = json.dumps(all_components_shap_values_instance, indent=4)
    with open('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine/xai/JSON/shap_values_instance.json', 'w') as f:
        f.write(json_data_instance)

if __name__ == "__main__":
        main()


