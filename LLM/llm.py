
from transformers import pipeline
import warnings
import requests
import os
import streamlit as st
import ollama

warnings.filterwarnings('ignore')
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

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

def shap_tranform(scenario):
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

Receive Input: You will receive SHAP values shap values used in the waterfall plot in the waterfall plot lower shapvalues are not shown and only the 10 features with high shap values are shown.
Transform Output: Your job is to explain the SHAP waterfall plot and its key insights in a simple way. Help the user understand:
            -- Which features contributed to a faulty prediction (positive SHAP values).
            -- Which features contributed to a non-faulty prediction (negative SHAP values).
            -- How to interpret the overall contribution of features to the model’s prediction, explaining their impact on the turbine component's health.
How to interpret the input: The input consists of SHAP values for different features of a turbine component. Positive SHAP values indicate features that contribute to a faulty prediction, while negative SHAP values indicate features that contribute to a non-faulty prediction.

Output Structure:

Explain about What is SHAP and waterfall plots of shap? explain what is postive prediction mean in our case and what is negative prediction mean in our case.
Explain the SHAP Waterfall Plot: Describe the SHAP waterfall plot and its components in simple terms. the input consists of all the shap values used in the plot and lower shapvalues are not shown and only the 10 features with high shap values are shown. 
Actionable Insights: Explain which factors are causing the prediction? 
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
    anchor =  {
        "id": 25,
        "component": "GEARBOX",
        "anchor": "generator_rotations_per_minute_min <= 0.00",
        "precision": 0.96,
        "coverage": 0.74,
        "prediction": "non-faulty"
    }
    shap = {"""generator_rotations_per_minute_min: 0.07361021637916565
                generator_bearing_temperature_average: 0.6213906407356262
                minimum_ambient_wind_speed: 0.014370696619153023
                average_absolute_ambient_wind_direction: -0.20554301142692566
                average_ambient_temperature: 0.5872780680656433
                average_controller_top_temperature: 0.04697530344128609
                average_controller_hub_temperature: 0.8406192064285278
                average_grid_rotor_inverter_phase_2_temperature: 0.2632294297218323
                average_grid_rotor_inverter_phase_3_temperature: 0.7883936166763306
                grid_production_frequency_average: -0.2571744918823242
                grid_production_power_maximum: 0.2717219293117523
                generator_bearing2_temperature_average: 1.1649558544158936
                max_windspeed3: 0.2713521420955658"""}
    counterfactual =  {
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
    }
    anchor_explanation = anchor_tranform(anchor)
    print(anchor_explanation)
    shap_explanation = shap_tranform(shap)
    print(shap_explanation)
    counterfactual_explanation = counterfactual_tranform(counterfactual)
    print(counterfactual_explanation)
    return anchor_explanation, shap_explanation, counterfactual_explanation

if __name__ == "__main__":
    main()
