
from transformers import pipeline
import warnings
import requests
import os
import streamlit as st
import ollama
import streamlit as st

warnings.filterwarnings('ignore')
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

@st.cache_resource
def anchor_tranform(scenario):
    prompt = f"""You are an AI assistant that transforms technical outputs from explainable AI (XAI) algorithms into a format that is easy for humans to understand. Your goal is to provide clear, concise, and informative explanations of XAI outputs, making them accessible to individuals without technical backgrounds.


We are analyzing data from a wind farm, focusing on the performance of five turbines (T01, T06, T07, T09, T11). Each turbine is equipped with various sensors that collect data every 10 minutes over two years. The dataset includes sensor data (81 variables like generator speed, pitch angle, wind speed), meteorological data (40 variables including temperature and humidity), and failure data for critical components (gearbox, transformer, generator bearing, generator, hydraulic group). Our machine learning model predicts whether a component is faulty or not based on this data.

Task:

Receive Input: You are given a scenario involving a specific turbine component (e.g., gearbox) a set of XAI anchor outputs indicating specific conditions or features that contributed to the model's prediction of a fault or non-fault in a turbine component.

Transform Output: Your job is to explain these conditions in a clear, non-technical way. Additionally, help users understand:
    --if the component is not faulty the explain the permissible changes they can make to maintain a non-failure prediction.
    --if the component is faulty then explain the adjustments they should make or which components to examine in order to alter a failure prediction and avoid future faults.


Output Structure:


Anchor Condition: Simplified explanation of the condition. Explanation of the precision value. Explanation of the coverage value.
Actionable Insights: For a non-faulty prediction, explain what changes are permissible to ensure the component remains non-faulty. For a faulty prediction, explain what adjustments could be made to potentially alter the failure prediction.
Conclusion: Provide a brief summary that helps the user understand the implications of the prediction and suggested actions.

Input:
{scenario}

Transformed Explanation:
"""
    response = ollama.chat(model='llama3', options={'temperature': 0}, messages=[{
        'role': 'user',
        'content': prompt,
    }])
    explanation = response['message']['content']
    return explanation

@st.cache_resource
def shap_tranform(scenario):
    prompt = f"""You are an intelligent assistant designed to explain complex machine learning predictions and their visualizations, like SHAP waterfall plots, in simple and easy-to-understand terms for users with no background in machine learning. Your main goal is to explain why the model made a specific prediction and provide actionable insights based on the SHAP values.


Context:

We are analyzing data from a wind farm, focusing on the performance of five turbines (T01, T06, T07, T09, T11). 
Each turbine is equipped with various sensors that collect data every 10 minutes over two years. 
The dataset includes sensor data (81 variables like generator speed, pitch angle, wind speed), meteorological data (40 variables including temperature and humidity), 
and failure data for critical components (gearbox, transformer, generator bearing, generator, hydraulic group).

Our machine learning model predicts whether a turbine component (such as the gearbox or generator bearing) is faulty or non-faulty. 
To make these predictions more understandable, we use SHAP (SHapley Additive exPlanations), 
a method that explains how much each feature in the data contributes to a particular prediction.


Task:

Receive Input: You will receive SHAP values shap values used in the waterfall plot in the waterfall plot lower shapvalues are not shown and only the 10 features with high shap values are shown.
Transform Output: Your job is to explain the SHAP waterfall plot and its key insights in a simple way. Help the user understand:
            -- Explain what is the prediction ML model made for the instance by considering target class input provided.  
            -- How to interpret the overall contribution of features to the model’s prediction, explaining their impact on the turbine component's health.
How to interpret the input: The input consists of SHAP values for different features of a turbine component. Positive SHAP values indicate features that contribute to a faulty prediction, while negative SHAP values indicate features that contribute to a non-faulty prediction. The input also consists of target class which specifies the prediction of the instance.

Output Structure:

1. Prediction: Explain the prediction made by the ML model for the instance based on the target class value provided. Describe what positive and negative predictions mean in this context.
2. Actionable Insights: Explain which factors are pushing the prediction of that instance by considering the target class value. If the prediction is non-faulty, explain 3 features with the highest negative SHAP values (i.e., the features that contributed the most to keeping the prediction non-faulty). If the prediction is faulty, explain 3 features with the highest positive SHAP values (i.e., the features that contributed the most to the failure prediction. Ensure that the explanation is based on the actual top 3 SHAP values provided.
3. Conclusion: Summarize the key insights from the explanation and provide recommendations if needed.

Explain all the above in 5-6 sentences following the output structure give the heading of which part of the output structure you are explaining. 


Input:
{scenario}

Transformed Explanation:
"""
    response = ollama.chat(model='llama3', options={'temperature': 0}, messages=[{
        'role': 'user',
        'content': prompt,
    }])
    explanation = response['message']['content']
    return explanation

@st.cache_resource
def counterfactual_tranform(scenario):
    prompt = f"""You are an intelligent assistant designed to explain complex machine learning predictions and their visualizations, like SHAP waterfall plots, in simple and easy-to-understand terms for users with no background in machine learning.


Context:

We are analyzing data from a wind farm, focusing on the performance of five turbines (T01, T06, T07, T09, T11). 
Each turbine is equipped with various sensors that collect data every 10 minutes over two years. 
The dataset includes sensor data (81 variables like generator speed, pitch angle, wind speed), meteorological data (40 variables including temperature and humidity), 
and failure data for critical components (gearbox, transformer, generator bearing, generator, hydraulic group).

Our machine learning model predicts whether a turbine component (such as the gearbox or generator bearing) is faulty or non-faulty. 
To make these predictions more understandable, we use SHAP (SHapley Additive exPlanations), 
a method that explains how much each feature in the data contributes to a particular prediction.


Task:

Receive Input: You will receive counterfactual suggestions showing which features and values should be adjusted to change a faulty prediction to non-faulty for a turbine component. 
Transform Output: Your job is to explain these counterfactual suggestions in a clear, non-technical way. Help users understand:
            -- Which components need to be examined and which features need to be adjusted to change the prediction from faulty to non-faulty.
            -- How much each feature needs to change to alter the prediction.
            -- Why these specific adjustments are important to improve the component's condition.
How to interpret the input: The input consists of counterfactual suggestions showing which features and values should be adjusted to change a faulty prediction to non-faulty for a turbine component. It also consists of the original value and the counterfcatual value. 

Output Structure:

Output Structure:

What are Counterfactuals? Provide a brief, easy-to-understand explanation of counterfactuals and their role in changing model predictions.
Necessary Adjustments: List the features that need to change, explaining how much they should be adjusted to switch the prediction from faulty to non-faulty, and which components to examine based on these adjustments.
Actionable Insights: Explain why adjusting these features or examining these components could improve the component’s health and reduce the likelihood of failure.
Conclusion: Summarize the key takeaways and how making these adjustments could prevent future faults.



Input:
{scenario}

Transformed Explanation:
"""
    response = ollama.chat(model='llama3', options={'temperature': 0}, messages=[{
        'role': 'user',
        'content': prompt,
    }])
    explanation = response['message']['content']
    return explanation


def main():
    '''anchor =  {
        "id": 25,
        "component": "GEARBOX",
        "anchor": "generator_rotations_per_minute_min <= 0.00",
        "precision": 0.96,
        "coverage": 0.74,
        "prediction": "non-faulty"
    }'''
    shap = [{"target_class":" Faulty"}, {
               "generator_rotations_per_minute_min": 0.646938145160675,
                "generator_bearing_temperature_average": -0.7935078740119934,
                "minimum_ambient_wind_speed": 0.011441145092248917,
                "average_absolute_ambient_wind_direction": 1.546500325202942,
                "average_ambient_temperature": 1.4875788688659668,
                "average_controller_top_temperature": -0.3365827202796936,
                "average_controller_hub_temperature": -0.3822290599346161,
                "average_grid_rotor_inverter_phase_2_temperature": 0.21342462301254272,
                "average_grid_rotor_inverter_phase_3_temperature": 2.0366032123565674,
                "grid_production_frequency_average": -0.11417296528816223,
                "grid_production_power_maximum": 0.10816410928964615,
                "generator_bearing2_temperature_average": -0.6663039326667786,
                "max_windspeed3": 0.4138451814651489
            }]
    '''counterfactual =  {
        "instance_index": 26,
        "changes": {
            "average_ambient_temperature": {
                "original": 24.1875,
                "counterfactual": 28.195341
            },
            "grid_production_power_maximum": {
                "original": 2099.5,
                "counterfactual": 76.8
            }
        },
        "component": "GEARBOX"
    }'''
    #anchor_explanation = anchor_tranform(anchor)
    #print(anchor_explanation)
    shap_explanation = shap_tranform(shap)
    print(shap_explanation)
    #counterfactual_explanation = counterfactual_tranform(counterfactual)
    #print(counterfactual_explanation)
    return shap_explanation

if __name__ == "__main__":
    main()
