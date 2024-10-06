# sample_pages.py
import streamlit as st
from datetime import datetime
import sys
import os
import plotly.express as px
sys.path.append(os.path.abspath('/app/utils'))
from streamlit.delta_generator import DeltaGenerator
import json
import shap

import matplotlib.pyplot as plt
#sys.path.append(os.path.abspath('/Users/supriyasindigerekumaraswmamy/Desktop/Thesis/wind_Turbine'))
from utils.helper import *

def shap_explainer(index, component,shap_data,model):
    shap_data = shap_data.to_frame().T.drop(['target_class', 'turbine_id', 'timestamp','index'], axis=1)
    shap_data = shap_data.astype(float)
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_values = explainer.shap_values(shap_data)
    shap_explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=shap_data)
    shap_explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=shap_data)
    fig, ax = plt.subplots(figsize=(100, 4))  # Set the desired size (width, height)
    shap.plots.waterfall(shap_explanation[0], show=False)  # Disable automatic display
    fig.set_size_inches(12, 5)  # Set the desired size (width, height)

    with open('./xai/JSON/shap_values_instance.json') as f:
        data = json.load(f)
        shap_value = data.get(component, {}).get(index)
    # Return the figure with the smaller plot
    return fig, shap_value

  
def get_explanation(timestamp, turbine_id, selected_features, component,model):
    data = selected_features
    data.reset_index(inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.date
    date_selected = pd.to_datetime(timestamp).date()

    # Find the correct index based on turbine_id and date_selected
    try:
        turbine_data_idx = data[(data['turbine_id'] == turbine_id) & (data['timestamp'] == date_selected)].index[0]
    except IndexError:
        st.write(f"No data found for turbine ID: {turbine_id} on date: {date_selected}")
        return
    shap_data = data.iloc[turbine_data_idx]
   
    turbine_data_idx = str(turbine_data_idx)  
    plot = shap_explainer(turbine_data_idx, component, shap_data,model)[0]
    shap_value = shap_explainer(turbine_data_idx, component, shap_data,model)[1]

    # Print the shap values index in the streamlit app
    st.write("SHAP Values index:")
    st.write(turbine_data_idx)

    # Print the shap values in the streamlit app
    st.write("SHAP Values:")
    st.write(shap_value)

 
    st.write("SHAP Waterfall Plot:")
    #fit the plot in the streamlit app for the conatiner
    st.pyplot(plot)