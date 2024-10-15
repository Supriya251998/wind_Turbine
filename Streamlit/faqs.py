import streamlit as st

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from ollama import Client
import ollama
import os
import requests


def generate_answer(question):
    ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    client = Client(host=ollama_host)

    prompt = f"""You are an AI assistant that answers questions in a clear, concise, and informative manner. Your goal is to provide helpful and accurate responses to user-submitted questions. Make sure to address the user's query directly and provide any necessary explanations.

Instructions:
- Answer the question thoroughly but in simple language.
- If the question is technical, simplify the explanation without losing accuracy.
- If needed, provide additional context to help the user understand the answer.
- Avoid overly technical jargon and focus on clarity.

User Question:
{question}

Answer:
"""
    response = ollama.chat(model='llama3', options={'temperature': 0}, messages=[{
        'role': 'user',
        'content': prompt,
    }])
    
    answer = response['message']['content']
    return answer

def show_faq():
        st.subheader("FAQs")
        st.write("Frequently Asked Questions about the Explainabl AI algorithm used in this system.")

        st.write("**Q1: What is SHAP? Which aspect of the model prediction does it explain**")
        st.write("Ans1: SHAP (SHapley Additive exPlanations) explains a machine learning model's prediction by assigning a score to each feature, showing how much it contributed. A high SHAP value means the feature strongly influenced the outcome.")
        
        st.write("**Q2: What is a SHAP Waterfall plot**")
        st.write("Ans2: A SHAP Waterfall plot visually shows how individual features impact a model’s prediction. Starting from a baseline value (the average prediction), it displays how each feature increases or decreases the prediction, with the most influential features shown first.")
        st.write("**Q3: What do negative and positive values represent in the SHAP waterfall plot?**")
        st.write("Ans3: In a SHAP waterfall plot, negative values depicted in blue encourage a non-faulty prediction, while positive values shown in red indicate a faulty prediction.")

        st.write("**Q4: What are Anchors?**")
        st.write("Ans4: Anchors explain predictions by identifying rules that anchor a decision. They offer clear, understandable explanations by showing which conditions or sensor value ranges must be met for the model to maintain a specific prediction.")

        st.write("**Q5: What are Anchor rules**")
        st.write("Ans5: Anchor rules are easy-to-understand, if-then conditions that explain how a machine learning model makes specific predictions. These rules must be met for the model to arrive at a particular outcome.")

        st.write("**Q5: What are Counterfactual Values?**")
        st.write("Ans5: Counterfactuals answer the question: **What needs to change for the outcome to be different?** Counterfactuals tell us what adjustments in sensor readings or other features would result in the wind turbine being predicted as **not faulty**.")

        st.write("**If you have any additional questions or need further clarification, feel free to ask in the chatbox. Your questions will be answered by llama3 , our advanced AI assistant.**")
        question = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            if question:
                answer = generate_answer(question)
                st.write(answer)
            else:
                st.write("Please enter a question.")

        