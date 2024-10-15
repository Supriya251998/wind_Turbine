# sample_pages.py
import streamlit as st
from datetime import datetime
import sys
import os
import plotly.express as px
#sys.path.append(os.path.abspath('/app/utils'))
from streamlit.delta_generator import DeltaGenerator
from explanation import *
sys.path.append(os.path.abspath('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine'))
from utils.helper import *

@st.cache_data
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
    return selected_features_data, components,models

def get_clean_rendering_container(app_state: str) -> DeltaGenerator:
    """Makes sure we can render from a clean slate on state changes."""
    slot_in_use = st.session_state.slot_in_use = st.session_state.get("slot_in_use", "a")
    if app_state != st.session_state.get("previous_state", app_state):
        if slot_in_use == "a":
            slot_in_use = st.session_state.slot_in_use = "b"
        else:
            slot_in_use = st.session_state.slot_in_use = "a"

    st.session_state.previous_state = app_state

    slot = {
        "a": st.empty(),
        "b": st.empty(),
    }[slot_in_use]

    return slot.container()

@st.cache_data
def get_turbine_data(selected_turbine_id, date_selected, selected_features):
    data = selected_features
    data['timestamp'] = data['timestamp'].dt.date
    date_selected = pd.to_datetime(date_selected).date()
    turbine_data = data[data['turbine_id'] == selected_turbine_id]
    turbine_data = turbine_data[turbine_data['timestamp'] < date_selected]
    turbine_data = turbine_data.sort_values(by='timestamp', ascending=False)
    turbine_data = turbine_data.head(7)
    turbine_data = turbine_data.sort_values(by='timestamp', ascending=True)
    # map the target class if it is 1 to faulty and 0 to not faulty
    turbine_data['target_class'] = turbine_data['target_class'].map({1: 'Faulty', 0: 'Not Faulty'})
    return turbine_data


def plot_turbine_prediction(selected_turbine_id, date_selected, selected_features, plot_container):
    # Clear the container before plotting new graph
    plot_container.empty() 
    
    # Get the last 7 instances of turbine data
    turbine_data = get_turbine_data(selected_turbine_id, date_selected, selected_features)
    
    if turbine_data.empty:
        
        plot_container.write(f"No data available for Turbine {selected_turbine_id} in the past 7 days for the prediction trend graph.")
        return
    
    fig = px.scatter(turbine_data, 
                     x='timestamp', 
                     y='target_class', 
                     title=f'7-Day Prediction Trend: Turbine {selected_turbine_id} Fault Status',
                     labels={'timestamp': 'Timestamp', 'target_class': 'Prediction'})
    
    fig.update_xaxes(tickvals=turbine_data['timestamp'], tickangle=45)
    
    # Display the updated plot in the specific container
    plot_container.plotly_chart(fig, use_container_width=True)



def render_turbine_page():
    selected_features, components,models = get_data()

 
    turbine_id_col = selected_features[components[0]][1]['turbine_id'].unique()  
    date_col = selected_features[components[0]][1]['timestamp'].dt.date
    date_col = date_col.sort_values().unique()

    turbine_id = st.selectbox("Select Turbine ID", turbine_id_col)

    if turbine_id:
        date = st.selectbox("Select Date", options=date_col)

        if date:
            st.markdown("<h3><b>Turbine Component Status</b></h3>", unsafe_allow_html=True)
            
            for i, component in enumerate(components):
                # Use an expander or a separate container for each component section
               if has_data(turbine_id, date, selected_features[component][1]):
                 #with st.expander(f"### **{component.replace('_', ' ').capitalize()}**", expanded=True):
                    # print the component
                    
                    st.markdown(f"<h4><b>{component.replace('_', ' ').capitalize()}</b></h4>", unsafe_allow_html=True)

                    # Display the fault status
                    component_faulty = is_faulty(turbine_id, date, selected_features[component][1])
                    if component_faulty:
                        st.error("Faulty")    
                    else:
                        st.success("Not Faulty")
                        
                    plot_container = get_clean_rendering_container(f"{turbine_id}_{component}")
                    plot_turbine_prediction(turbine_id, date, selected_features[component][1], plot_container)
                    # Display the explanation if the button was clicked
                    st.markdown("<h5><b>Explanation Methods for the model prediction</b></h3>", unsafe_allow_html=True)
                    explanation_method = st.selectbox("Select the below methods", 
                                                      ["None","Shap Explanation", "Anchor Explanation", "Counterfactual Explanation"],key=f"explanation_{component}")
                    if explanation_method != "None":
                        if explanation_method == "Shap Explanation":
                            shap_explainer(date, turbine_id, selected_features[component][1], component, models[component])
                        elif explanation_method == "Anchor Explanation":
                            anchor_explainer(date, turbine_id, selected_features[component][1], component)
                        elif explanation_method == "Counterfactual Explanation":
                            counterfactual_explainer(date, turbine_id, selected_features[component][1])


               else:
                    st.write(f"**{component.replace('_', ' ').capitalize()}**")
                    st.write(f"The data for the selected turbine on this date is not accessible.")
     
                       
def has_data(turbine_id, date_selected, component_data):
    # Filter the data for the selected turbine and date
    matching_data = component_data[(component_data['turbine_id'] == turbine_id) & 
                                   (component_data['timestamp'].dt.date == date_selected)]
    return not matching_data.empty


def is_faulty(turbine_id, date_selected, component_data):
    matching_data = component_data[(component_data['turbine_id'] == turbine_id) & 
                                   (component_data['timestamp'].dt.date == date_selected)]
    
    # Check if there's any data for the given turbine and date
    if matching_data.empty:
        return None  # No data available for this component
    
    # Check if any of the rows indicate a fault (target_class == 1)
    for record in matching_data.to_dict('records'):
        if record['target_class'] == 1:  # Fault detected
            return True

    return False
