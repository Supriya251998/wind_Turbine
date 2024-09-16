import warnings
warnings.filterwarnings("ignore")
import sys
import os
import dice_ml
from dice_ml.utils import helpers 
sys.path.append(os.path.abspath('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine'))
from utils.helper import *
import json
import pandas as pd

def get_data():
    failures = load_failures_data('./data/model_data/failures.csv')
    components = failures['component'].unique()
    component_data = load_all_component_data(components)
    data_splits = prepare_all_data_for_training(component_data, "target_class") [0]
    models = load_all_models(components, "xgb")
    selected_features_data = fit_and_select_features(models, data_splits)
    models = retrain_models_on_selected_features(models, selected_features_data, data_splits)

    return models, selected_features_data, components,data_splits

def counterfactuals():
    models, selected_features_data, components, data_splits = get_data()

    # Dictionary to store all counterfactuals
    all_counterfactuals = []

    for component in components:
        selected_features_data[component][0]['target_class'] = data_splits[component][2]
        continuous_features = list(selected_features_data[component][2])  # Ensure continuous features are provided as a list
        d = dice_ml.Data(dataframe=selected_features_data[component][0], continuous_features=continuous_features, outcome_name='target_class')
        m = dice_ml.Model(model=models[component], backend='sklearn')
        exp = dice_ml.Dice(d, m)
        
        for i in range(len(selected_features_data[component][1])):
            query_instance = selected_features_data[component][1].iloc[i]
            query_instance_df = pd.DataFrame(query_instance).T
            
            if models[component].predict(query_instance_df) == 1:
                dice_exp = exp.generate_counterfactuals(query_instance_df, total_CFs=1, desired_class="opposite")
                counterfactual_df = dice_exp.visualize_as_dataframe(show_only_changes=True)
                print("Counterfactuals for instance and component", i, component)
                # Extract counterfactual data
                counterfactual_example = dice_exp.cf_examples_list[0]
                original_values = query_instance_df.iloc[0]
                counterfactual_values = counterfactual_example.final_cfs_df.iloc[0]

                changes = {}
                for feature in original_values.index:
                    if original_values[feature] != counterfactual_values[feature]:
                        changes[feature] = {
                            'original': original_values[feature],
                            'counterfactual': counterfactual_values[feature]
                        }

                # Append changes to the list for the current component
                if changes:
                    all_counterfactuals.append({
                        'instance_index': i,
                        'changes': changes,
                        'component': component
                    })

    # Save all counterfactual changes to a single JSON file
    with open('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine/xai/JSON/counterfactual.json', 'w') as f:
        json.dump(all_counterfactuals, f, indent=4)

if __name__ == "__main__":
    counterfactuals()