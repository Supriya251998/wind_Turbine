# sample_pages.py
import streamlit as st
from datetime import datetime
import sys
import os
import plotly.express as px

from streamlit.delta_generator import DeltaGenerator
import json
import shap
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('/app/utils'))
#sys.path.append(os.path.abspath('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine'))
from helper import *
from llm import *

def check_condition(condition, data_instance):

    """
    Evaluates a single condition from the anchor rule against the input data instance.
    Returns True if the condition is satisfied, False otherwise.
    """
    try:
        return eval(condition, {}, data_instance)
    except Exception as e:
        return False

def display_decision_table_with_satisfaction(anchor_data, data_instance):
    """
    Function to display a decision table for anchor rules with satisfaction check in a Streamlit app.

    Parameters:
    anchor_data (dict): Dictionary containing anchor rule information.
    data_instance (dict): Dictionary containing input data for an instance.
    """
    # Parse the anchor rule into individual conditions
    conditions = anchor_data["anchor"].split(" AND ")

    # Check if each condition is satisfied
    satisfied_status = [check_condition(cond, data_instance) for cond in conditions]

    # Create a decision table for the conditions
    data = {
        "Condition": conditions,
        "Satisfied?": ["Yes" if status else "No" for status in satisfied_status],
        "Leads to": ["Proceed to next condition"] * (len(conditions) - 1) + [f'Prediction: {anchor_data["prediction"]}']
    }

    # Create a Pandas DataFrame for the table
    df = pd.DataFrame(data)

    return df

@st.cache_data
def get_index(timestamp, turbine_id, selected_features):
    data = pd.DataFrame(selected_features)
    data.reset_index(inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.date
    data['target_class'] = data['target_class'].map({1: 'Faulty', 0: 'Not Faulty'})
    date_selected = pd.to_datetime(timestamp).date()


    # Find the correct index based on turbine_id and date_selected
    try:
        turbine_data_idx = data[(data['turbine_id'] == turbine_id) & (data['timestamp'] == date_selected)].index[0]
    except IndexError:
        st.write(f"No data found for turbine ID: {turbine_id} on date: {date_selected}")
        return
    
    shap_data = data.iloc[turbine_data_idx]

    return shap_data, turbine_data_idx

@st.cache_resource
def shap_explainer(timestamp, turbine_id, selected_features, component, _model):
    shap_data, index = get_index(timestamp, turbine_id, selected_features)
    
    target_class = shap_data['target_class']
    shap_data = shap_data.to_frame().T.drop(['target_class', 'turbine_id', 'timestamp','index'], axis=1)
    shap_data = shap_data.astype(float)
    explainer = shap.TreeExplainer(_model.named_steps['model'])
    shap_values = explainer.shap_values(shap_data)
    shap_explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=shap_data)
    shap_explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=shap_data)
    fig, ax = plt.subplots(figsize=(100, 4))  # Set the desired size (width, height)
    shap.plots.waterfall(shap_explanation[0], show=False)  # Disable automatic display
    fig.set_size_inches(10, 8)  # Set the desired size (width, height)
    
    index = str(index)
    with open('./xai/JSON/shap_values_instance.json') as f:
        data = json.load(f)
        shap_value = data.get(component, {}).get(index)
        shap_value = list(shap_value.values())
        shap_value.insert(0, {"target_class": target_class})
    
    # get the explanation of the waterfall plot from llm,
    llm_explanation = shap_tranform(shap_value)
    
    
    st.markdown("<h4><b>Understanding the model prediction using SHAP</b></h4>", unsafe_allow_html=True)  
    st.write('''Using SHAP, we aim to provide clear insights into the factors influencing the model's predictions, helping users understand the reasoning behind every prediction.''')
    st.markdown(
    '<h5><b>SHAP Waterfall Plot</b> <a href="https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html" target="_blank">?</a></h5>',
    unsafe_allow_html=True)

    
    #st.write('''The SHAP waterfall plot below illustrates how each feature contributes to the final prediction, aiding in the understanding of each feature's impact on the model's decision. 
     #           In the plot, negative values depicted in blue encourage a non-faulty prediction, while positive values shown in red indicate a faulty prediction.''')
    st.pyplot(fig)

    st.markdown("<h5><b>SHAP waterfall plot explanation</b></h3>", unsafe_allow_html=True)
    
    st.write(llm_explanation)
   

@st.cache_resource
def anchor_explainer(timestamp, turbine_id, selected_features, component):
    shap_data, index = get_index(timestamp, turbine_id, selected_features)
    
    

    with open('./xai/JSON/anchors.json') as f:
        data = json.load(f)
        
    for entry in data:
        if entry['id'] == index and entry['component'] == component:
            anchor_value = entry
            break
    decison_table = display_decision_table_with_satisfaction(anchor_value, shap_data)
    anchor_explanation = anchor_tranform(anchor_value)
    # Display the decision table in the app
    if shap_data['target_class'] == 'Faulty':
        st.markdown("<h4><b>Conditions Leading to a Faulty Prediction</b></h4>", unsafe_allow_html=True)  
        st.write("The anchor rule below outlines the critical conditions that lead to a faulty prediction. We present the anchor rules in the decision table below and provide an explanation of this specific rule as well.")

    else:
        st.markdown("<h4><b>Permissible Changes to Maintain a Non-Failure Prediction</b></h4>", unsafe_allow_html=True)  
        st.write("The anchor rule below outlines the conditions that must be met to sustain a non-faulty prediction. We present the anchor rules in the decision table below and offer an explanation of this specific rule as well.")
    shap_data = shap_data.to_frame().T.drop(['target_class', 'turbine_id', 'timestamp','index'], axis=1)
    st.markdown("<h5><b>Original Data Instance</b></h3>", unsafe_allow_html=True)
    st.dataframe(shap_data,width=1000)
    st.markdown("<h5><b>Anchor Rule Decision Table with Satisfaction</b></h3>", unsafe_allow_html=True)
    st.write("The decision table below shows the **Condition** (the specific criteria linked to a fault), whether this condition was **Satisfied** (met by the current data), and **Leads To** (the outcome if the condition is met, such as a fault prediction).")
    st.dataframe(decison_table, width=1000)
    st.markdown("<h5><b>Anchor Rule Explanation</b></h3>", unsafe_allow_html=True)
    st.write(anchor_explanation)

@st.cache_resource
def counterfactual_explainer(timestamp, turbine_id, selected_features):
    shap_data, index = get_index(timestamp, turbine_id, selected_features)
    if shap_data['target_class'] == 'Faulty':
        shap_data = shap_data.to_frame().T.drop(['target_class', 'turbine_id', 'timestamp','index'], axis=1)
        with open('./xai/JSON/counterfactual.json') as f:
            data = json.load(f)

        for entry in data:
            if entry["instance_index"] == index:
                counterfactual_value = entry
                break

        changes = counterfactual_value["changes"]
        counterfactual_df = pd.DataFrame({
            "Feature": list(changes.keys()),
            "Counterfactual Value": [changes[feature]["counterfactual"] for feature in changes],
        })
        counterfactual_explanation = counterfactual_tranform(counterfactual_value)

        st.markdown("<h4><b>Adjustments to Modify Failure Predictions</b></h4>", unsafe_allow_html=True)  
        st.write("The counterfactual explanation below outlines the changes required to transform a faulty prediction into a non-faulty one. We present the original data along with counterfactual changes in the table below.")
        st.markdown("<h5><b>Original Data with faulty prediction</b></h3>", unsafe_allow_html=True)
        st.dataframe(shap_data,width=1000)
        st.markdown("<h5><b>Counterfactual Data with non faulty prediction</b></h3>", unsafe_allow_html=True)
        st.dataframe(counterfactual_df,width=1000)
        
        st.markdown("<h5><b>Counterfactual Explanation</b></h3>", unsafe_allow_html=True)
        st.write(counterfactual_explanation)
    else:
        st.write("The selected instance is not faulty, hence no counterfactual explanation is available.") 

    

  

  
    