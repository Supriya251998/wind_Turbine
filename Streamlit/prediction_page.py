# sample_pages.py
import streamlit as st
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine '))
from helper import *


def get_data():
    failures = load_failures_data('./data/model_data/failures.csv')
    components = failures['component'].unique()
    component_data = load_all_component_data(components)
    data_splits, ui_data_splits = prepare_all_data_for_training(component_data, "target_class")
    models = load_all_models(components, "xgb")
    selected_features_data = fit_and_select_features(models, data_splits)
    models = retrain_models_on_selected_features(models, selected_features_data, data_splits)
    
    for component in components:
        # Predict and assign the class label
        y_predict = models[component].predict(selected_features_data[component][1])
        selected_features_data[component][1]['target_class'] = y_predict
        selected_features_data[component][1]['target_class'] = selected_features_data[component][1]['target_class']
        
        # Assign turbine_id and convert timestamps
        selected_features_data[component][1]['turbine_id'] = ui_data_splits[component][1]['turbine_id']
        ui_data_splits[component][1]['timestamp'] = pd.to_datetime(ui_data_splits[component][1]['timestamp'])
        
        
        # Convert to datetime and extract the date
        selected_features_data[component][1]['timestamp'] = ui_data_splits[component][1]['timestamp']
    return selected_features_data, components


def render_turbine_page():
    selected_features, components = get_data()
    
    # Assuming same turbine_ids across components
    turbine_id_col = selected_features[components[0]][1]['turbine_id'].unique()  
    min_date = datetime(2016, 1, 1)
    max_date = datetime(2017, 12, 31)
    
    # Step 1: User selects Turbine ID first
    turbine_id = st.selectbox("Select Turbine ID", turbine_id_col)
    
    # Step 2: Only show the date selection after a turbine ID is selected
    if turbine_id:
        date = st.date_input("Select Date", min_value=min_date, max_value=max_date)
        
        # Step 3: Only show fault status after both turbine ID and date are selected
        if date:
            st.write("### Turbine Component Statuses")
            cols = st.columns(len(components))  # Create one column per component

            # Check each component dataset for faults and display in the respective column
            for i, component in enumerate(components):
                component_faulty = is_faulty(turbine_id, date, selected_features[component][1])
                
                # Use a more compact label if necessary
                display_name = component.replace("_", " ")  # Replaces underscores with spaces for better display
                
                with cols[i]:  # Each component's status is displayed in its respective column
                    st.write(f"**{display_name.capitalize()}**")
                    if component_faulty:
                        st.error(f"Faulty")
                    else:
                        st.success(f"Not Faulty")
            
            st.write("### Details")
            st.write("Here you can see the details of the turbine components.")
            
            # Additional details for faulty components
            for component in components:
                component_faulty = is_faulty(turbine_id, date, selected_features[component][1])
                if component_faulty:
                    st.write(f"Content from OpenAI for {component} fault detected.")

def is_faulty(turbine_id, date_selected, component_data):
    # Convert component data to records for iteration
    for record in component_data.to_dict('records'):
        # record['timestamp'] is already a pandas.Timestamp, so no need to convert
        record_timestamp = record['timestamp'].date()  
        
        # Check if turbine_id and date match, and if target_class is 1 (indicating a fault)
        if (record['turbine_id'] == turbine_id and 
            record_timestamp == date_selected and 
            record['target_class'] == 1):
            return True
    return False
