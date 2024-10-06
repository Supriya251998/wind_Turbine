# sample_pages.py
import streamlit as st
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath('/app/utils'))
from utils.helper import *


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
    date_col = selected_features[components[0]][1]['timestamp'].dt.date
    date_col = date_col.sort_values().unique()
    min_date = datetime(2016, 1, 1)
    max_date = datetime(2017, 12, 31)
    
    # Step 1: User selects Turbine ID first
    turbine_id = st.selectbox("Select Turbine ID", turbine_id_col)
    
    # Step 2: Only show the date selection after a turbine ID is selected
    if turbine_id:
        date = st.selectbox("Select Date", options=date_col)
        
        # Step 3: Only show fault status after both turbine ID and date are selected
        if date:
            st.write("### Turbine Component Statuses")
            cols = st.columns(len(components))  # Create one column per component
            buttons_clicked = {}

            # Check each component dataset for faults and display in the respective column
            for i, component in enumerate(components):
                component_faulty = is_faulty(turbine_id, date, selected_features[component][1])
                
                # Use a more compact label if necessary
                display_name = component.replace("_", " ")  # Replaces underscores with spaces for better display
                with cols[i]:  # Each component's status is displayed in its respective column
                    st.write(f"**{display_name.capitalize()}**")
                    if component_faulty:
                        # Simulate a red background for faulty status
                        st.error("Faulty")
                        # Create a button with a unique key
                        if st.button("Why?", key=f"why_faulty_{component}"):
                            buttons_clicked[component] = "Faulty"
                    else:
                        # Simulate a green background for non-faulty status
                        st.success("Not Faulty")
                        # Create a button with a unique key
                        if st.button("Why?", key=f"why_not_faulty_{component}"):
                            buttons_clicked[component] = "Not Faulty"
            
            for component in components:
                if component in buttons_clicked:
                    st.write("### Explanations")
                    status = buttons_clicked[component]
                    if status == "Faulty":
                        st.write(f"Explanation for {component} fault: Content from OpenAI.")
                    else:
                        st.write(f"{component.capitalize()} is operating normally.")

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






def get_turbine_data(selected_turbine_id, date_selected, selected_features):
    data = selected_features
    
    # Convert timestamp to datetime format (if not already)
    data['timestamp'] = data['timestamp'].dt.date
    
    # Convert date_selected to datetime and set timezone to UTC
    date_selected = pd.to_datetime(date_selected)
    
    # Filter data for the selected turbine
    turbine_data = data[data['turbine_id'] == selected_turbine_id]
    
    # Filter data for all instances before the selected date
    turbine_data = turbine_data[turbine_data['timestamp'] < date_selected]
    
    # Sort by timestamp in descending order to get the most recent entries
    turbine_data = turbine_data.sort_values(by='timestamp', ascending=False)
    
    # Select the last 7 instances (rows)
    turbine_data = turbine_data.head(7)
    
    # Sort again in ascending order to preserve the original order of time
    turbine_data = turbine_data.sort_values(by='timestamp', ascending=True)
    
    return turbine_data

def plot_turbine_prediction(selected_turbine_id, date_selected,selected_features,plot_container):
    # Get the last 7 instances of turbine data
    plot_container.empty() 
    turbine_data = get_turbine_data(selected_turbine_id, date_selected,selected_features)
    
    # Check if there are any data points
    if turbine_data.empty:
        print("No data available for the selected turbine and date.")
        return
    
    # Create a scatter plot using Plotly
    
        
    fig = px.scatter(turbine_data, 
                     x='timestamp', 
                     y='target_class', 
                     title=f'Scatter Plot of Prediction for Turbine {selected_turbine_id} (Last 7 Instances)',
                     labels={'timestamp': 'Timestamp', 'target_class': 'Prediction'})
    
    fig.update_xaxes(tickvals=turbine_data['timestamp'], tickangle=45)
    # Show the plot
    plot_container.plotly_chart(fig, use_container_width=True)



def render_turbine_page():
    selected_features, components = get_data()
    
    # Assuming same turbine_ids across components
    turbine_id_col = selected_features[components[0]][1]['turbine_id'].unique()  
    date_col = selected_features[components[0]][1]['timestamp'].dt.date
    date_col = date_col.sort_values().unique()
    min_date = datetime(2016, 1, 1)
    max_date = datetime(2017, 12, 31)
    
    # Step 1: User selects Turbine ID first
    turbine_id = st.selectbox("Select Turbine ID", turbine_id_col)
    
    # Step 2: Only show the date selection after a turbine ID is selected
    if turbine_id:
        date = st.selectbox("Select Date", options=date_col)
        
        # Step 3: Only show fault status after both turbine ID and date are selected
        if date:
            st.write("### Turbine Component Status")
            cols = st.columns(len(components))  # Create one column per component
            if "graph_displayed" in st.session_state:
                del st.session_state["graph_displayed"]
            buttons_clicked = {}

            # Check each component dataset for faults and display in the respective column
            for i, component in enumerate(components):
                component_faulty = is_faulty(turbine_id, date, selected_features[component][1])
                
                # Use a more compact label if necessary
                # Replaces underscores with spaces for better display and make it bold

                display_name = component.replace("_", " ")  # Replaces underscores with spaces for better display
                  # Each component's status is displayed in its respective column in bold

                #st.write(f"###{display_name.capitalize()}")
                st.write(f"### **{display_name.capitalize()}**")

                #st.markdown(f"<h4 style='color: black; font-weight: bold;'>{display_name.capitalize()}</h4>", unsafe_allow_html=True)
                plot_containers = {component: st.empty() for component in components}
                if component_faulty:
                        # Simulate a red background for faulty status
                        st.error("Faulty")
                        # Create a button with a unique key
                        if st.button("Why?", key=f"why_faulty_{component}"):
                            buttons_clicked[component] = "Faulty"
                        
                    
                        plot_turbine_prediction(turbine_id, date, selected_features[component][1], plot_containers[component])
                else:
                        # Simulate a green background for non-faulty status
                        st.success("Not Faulty")
                        # Create a button with a unique key
                        if st.button("Why?", key=f"why_not_faulty_{component}"):
                            buttons_clicked[component] = "Not Faulty"
                        
                        plot_turbine_prediction(turbine_id, date, selected_features[component][1], plot_containers[component])
            
            for component in components:
                if component in buttons_clicked:
                    st.write("### Explanations")
                    status = buttons_clicked[component]
                    if status == "Faulty":
                        st.write(f"Explanation for {component} fault: Content from OpenAI.")
                    else:
                        st.write(f"{component.capitalize()} is operating normally.")


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


def plot_turbine_prediction(selected_turbine_id, date_selected, selected_features, plot_container):
    # Get the last 7 instances of turbine data
    plot_container.empty() 
    turbine_data = get_turbine_data(selected_turbine_id, date_selected, selected_features)
    
    # Check if there are any data points
    if turbine_data.empty:
        st.write("No data available for the selected turbine and date.")
        return
    
    # Initialize fig outside the loop
    fig = None
    
    # Create a scatter plot using Plotly
    for record in turbine_data.to_dict('records'):
        # Check if the turbine_id and timestamp match the selected va
        print(record['timestamp'])
        if (record['turbine_id'] == selected_turbine_id and 
            record['timestamp'] == date_selected):
            
            # If the condition is met, assign the figure
            fig = px.scatter(turbine_data, 
                     x='timestamp', 
                     y='target_class', 
                     title=f'Scatter Plot of Prediction for Turbine {selected_turbine_id} (Last 7 Instances)',
                     labels={'timestamp': 'Timestamp', 'target_class': 'Prediction'})
    
            fig.update_xaxes(tickvals=turbine_data['timestamp'], tickangle=45)
    
    # Ensure fig is not None before trying to plot it
    if fig is not None:
        plot_container.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No matching data for plotting.")




def render_turbine_page():
    selected_features, components = get_data()
    
    # Assuming same turbine_ids across components
    turbine_id_col = selected_features[components[0]][1]['turbine_id'].unique()  
    date_col = selected_features[components[0]][1]['timestamp'].dt.date
    date_col = date_col.sort_values().unique()
    min_date = datetime(2016, 1, 1)
    max_date = datetime(2017, 12, 31)
    
    # Step 1: User selects Turbine ID first
    turbine_id = st.selectbox("Select Turbine ID", turbine_id_col)
    
    # Step 2: Only show the date selection after a turbine ID is selected
    if turbine_id:
        date = st.selectbox("Select Date", options=date_col)
        
        # Step 3: Only show fault status after both turbine ID and date are selected
        if date:
            st.write("### Turbine Component Status")
            cols = st.columns(len(components))  # Create one column per component
            if "graph_displayed" in st.session_state:
                del st.session_state["graph_displayed"]
            buttons_clicked = {}

            # Check each component dataset for faults and display in the respective column
            for i, component in enumerate(components):
                component_faulty = is_faulty(turbine_id, date, selected_features[component][1])
                
                # Use a more compact label if necessary
                # Replaces underscores with spaces for better display and make it bold

                display_name = component.replace("_", " ")  # Replaces underscores with spaces for better display
                  # Each component's status is displayed in its respective column in bold

                #st.write(f"###{display_name.capitalize()}")
                st.write(f"### **{display_name.capitalize()}**")

                #st.markdown(f"<h4 style='color: black; font-weight: bold;'>{display_name.capitalize()}</h4>", unsafe_allow_html=True)
                plot_containers = {component: st.empty() for component in components}
                if component_faulty:
                        # Simulate a red background for faulty status
                        st.error("Faulty")
                        # Create a button with a unique key
                        if st.button("Why?", key=f"why_faulty_{component}"):
                            buttons_clicked[component] = "Faulty"
                        
                    
                        plot_turbine_prediction(turbine_id, date, selected_features[component][1], plot_containers[component])
                else:
                        # Simulate a green background for non-faulty status
                        st.success("Not Faulty")
                        # Create a button with a unique key
                        if st.button("Why?", key=f"why_not_faulty_{component}"):
                            buttons_clicked[component] = "Not Faulty"
                        
                        plot_turbine_prediction(turbine_id, date, selected_features[component][1], plot_containers[component])
            
            for component in components:
                if component in buttons_clicked:
                    st.write("### Explanations")
                    status = buttons_clicked[component]
                    if status == "Faulty":
                        st.write(f"Explanation for {component} fault: Content from OpenAI.")
                    else:
                        st.write(f"{component.capitalize()} is operating normally.")



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