from alibi.explainers import AnchorTabular
import time
from sklearn.model_selection import  train_test_split
import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.abspath('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine'))
from utils.helper import *

# Load the data
def get_data():
    failures = load_failures_data('./data/model_data/failures.csv')
    components = failures['component'].unique()
    component_data = load_all_component_data(components)
    data_splits = prepare_all_data_for_training(component_data, "target_class")
    models = load_all_models(components, "xgb")
    selected_features_data = fit_and_select_features(models, data_splits)
    models = retrain_models_on_selected_features(models, selected_features_data, data_splits)

    return models, selected_features_data, components

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  return "Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec)

def anchors_explanation():
    models, selected_features_data, components = get_data()
    anchors_list = []
    for component in components:
        globals()[f"{component}_predit_fn"] = lambda x: models[component].predict(x)
        globals()[f"{component}_explainer"] = AnchorTabular(globals()[f"{component}_predit_fn"], selected_features_data[component][2])
        globals()[f"{component}_explainer"].fit(selected_features_data[component][0].to_numpy())

        globals()[f"{component}_test_np"] = selected_features_data[component][1].to_numpy()
        globals()[f"{component}_test_length"] = globals()[f"{component}_test_np"].shape[0]
        globals()[f"anchors_{component}"] =[ ]

        start_time = time.time()
        for i in range(globals()[f"{component}_test_length"]):
            explanation = globals()[f"{component}_explainer"].explain(globals()[f"{component}_test_np"][i],threshold=0.95)
            globals()[f"anchors_{component}"].append(explanation)
            current_time = time.time()
            time_lapsed = current_time - start_time
            print("{} von {} Schritten abgeschlossen. Zeit: {}".format(i, globals()[f"{component}_test_length"], time_convert(time_lapsed)), end="\r", flush=True)
        print(f"{component} done")
        print(time_convert(time.time() - start_time))
        for index , elem in enumerate(globals()[f"anchors_{component}"]):
            anchor = ' AND '.join(elem["data"]["anchor"])  
            precision = float(elem["data"]["precision"])
            coverage = float(elem["data"]["coverage"])
            class_mapping = {i: class_label for i, class_label in enumerate(models[component].classes_)}
            class_mapping = {0: "non-faulty", 1: "faulty"}
            predition=int(elem['data']['raw']['prediction'])
            actual_prediction = class_mapping.get(predition,'Unknown')
            anchor_data = {
                "id": index,
                "component": component,
                "anchor": anchor,
                "precision": round(precision, 2),
                "coverage": round(coverage, 2),
                "prediction": actual_prediction,}
            anchors_list.append(anchor_data)
    import json
    with open('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine/xai/JSON/anchors.json', 'w') as json_file:
        json.dump(anchors_list, json_file, indent=4)


if __name__ == "__main__":
    anchors_explanation()
