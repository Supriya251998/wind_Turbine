import streamlit as st
from sample_pages import show_sample_page_1, show_sample_page_2
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import ollama

mlflow.set_tracking_uri("http://localhost:80") 

def get_mlflow_experiment_data(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        st.error(f"Experiment '{experiment_name}' not found.")
        return None

    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    # Extract metrics and parameters for display
    data = []
    for run in runs:
        data.append({
            "Run ID": run.data.tags["mlflow.runName"],
            "F1 Score": run.data.metrics.get("score"),
            "Hyperparameters": run.data.params
        })

    return pd.DataFrame(data)

def generate_answer(question):
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

def main():
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox("Select Page", ["Introduction", "Sample Page 1", "Sample Page 2"])

    if page == "Introduction":
        show_introduction()
    elif page == "Sample Page 1":
        show_sample_page_1()
    elif page == "Sample Page 2":
        show_sample_page_2()



def show_introduction():
    st.title("Introduction")
    #st.write("Welcome to the Introduction page about our model.")
    st.write(f'''The energy sector is currently undergoing a significant transformation, with wind energy standing out as the most advanced renewable technology globally. 
                 To maximize the profitability of wind turbines, it is crucial to implement predictive maintenance strategies. 
                 These strategies will help optimize energy production and minimize unexpected downtimes.''')
    st.write(f'''To achieve this we are developing a smart system that uses machine learning for predictive maintenance of wind turbines. 
                 We use an open source wind turbine dataset published by EDP to develop our predictive maintenance system. In the below following sections you can find details about the dataset, ML model used and its performance and Frequencly asked questions.''')
    # Sub-menu for Introduction
    intro_page = st.selectbox("Select a section", ["Overview of the Dataset", "Overview of the ML model", "FAQs"])

    if intro_page == "Overview of the Dataset":
        st.subheader("Dataset Overview")
        
        st.write('''This dataset, used in a 2019 competition and provided by EDP, includes Supervisory Control and Data Acquisition (SCADA) data from a wind farm. 
                 It contains measurements from 5 turbines labeled as T01, T06, T07, T09, and T11. 
                 The main focus of the dataset is monitoring the performance of key components such as the gearbox, generator, generator bearings, transformer, and hydraulic group.''')
        st.write("""Each turbine is equipped with sensors that collect data over a span of **two years**, 
                 with measurements recorded every **10 minutes**. These sensors track various performance parameters such as **generator bearing temperatures, 
                 oil temperature, pitch angle**, and **generator RPM**""")
        
        st.write("The dataset comprises of three main sections:")

        st.subheader("1. Sensor Data")
        st.write(""" This data contains **81 variables** including measurements of **generator speed, blade pitch angle, wind speed**, and **wind direction**.
                     The data was recorded every 10 minutes for each turbine, with the **standard deviation, minimum, and maximum** values of these variables stored.""")
        st.subheader("2. Meteorological Mast Data")
        st.write(""" This data captures the environmental conditions of the wind farm, including **40 variables** related to **Wind speed, Direction, Temperature, Pressure, Humidity and
                 Precipitation** """)
        st.subheader("3. Failure Data")
        st.write("""This section of the dataset records failure events for 5 critical components: **Gearbox, Transformer, Generator Bearing, Generator, Hydraulic Group**.
                 It contains details about the time, day, and characteristics of each failure, as well as which turbine experienced the failure.""")
        st.write("For more details on the dataset used, you can refer to the following papers and resources:")
        st.markdown("- [A Data-Centric Machine Learning Methodology: Application on Predictive Maintenance of Wind Turbines](https://www.mdpi.com/1996-1073/15/3/826)")
        
    elif intro_page == "Overview of the ML model":
        st.write("For implementing a Machine Learning model for our predictive maintenance system, we first preprocess the dataset and then use it to train the model.")
        
        st.subheader("Data Preprocessing")
        st.write("The dataset undergoes thorough preprocessing to ensure quality input for the machine learning model. ")
        st.markdown("- First, the signal and meteorological data from 2016 and 2017 are aggregated on a daily basis, with specific rules for minimum, maximum, and average values. ")
        st.markdown("-  Low-variance columns are removed, and missing values are handled through binary indicator columns and backward filling.")
        st.markdown("-  The cleaned signal and meteorological data are merged, feature names are expanded for clarity, and failure data is integrated to generate a binary target variable that indicates failure events.")
        st.markdown("-  At the end we get 5 data frames for each component, these data frames consists of signal data, meteorological data, Binary target variable, and the component feature that indicates the component at fault.")
        
        st.subheader("Machine Learning Model")
        st.write("We use a XGBoost classifier to predict failures in wind turbines based on the preprocessed dataset. ")
        st.markdown("- Preprocessed data for each component is split into 70% training and 30% testing datasets, and various hyperparameter configurations are explored in an Optuna study to maximize model performance.")
        st.markdown("- After optimization, the best-performing model for each component is logged and stored using MLflow.")
        
        st.subheader("Model Performance")
        st.write(" We implement the above mentioned XGBoost Classfication model for each of the components and the model's performance is evaluated using the F1 score, which balances precision and recall to provide a comprehensive measure of the model's effectiveness.")
        experiment_name = "predictive_maintenance_system"
        experiment_data = get_mlflow_experiment_data(experiment_name)
    # Visualize the F1 score using a Plotly Bar Chart
        fig = px.bar(
            experiment_data, 
            x="Run ID", 
            y="F1 Score", 
            title="F1 Score Across Components", 
            labels={"F1 Score": "F1 Score", "Run ID": "Component"},
            height=400
        )
        st.plotly_chart(fig)
       
    elif intro_page == "FAQs":
        st.subheader("FAQs")
        st.write("Frequently Asked Questions about the predictive Maintenance System.")

        st.write("**Q1: What is predictive maintenance, and why is it important for wind turbines?**")
        st.write("Ans1: Predictive maintenance uses data from wind turbines to predict when a failure might happen. This helps in fixing issues before they cause unexpected downtime, saving time and money.")

        st.write("**Q2: Where does the data used in this system come from?**")
        st.write("Ans2: The data comes from sensors placed on wind turbines, tracking things like temperature, speed, and other conditions. It was collected by EDP during a competition in 2019.")

        st.write("**Q3: Which parts of the wind turbines are monitored by this system?**")
        st.write("Ans3: The system monitors important parts of the turbines like the gearbox, generator, bearings, transformer, and hydraulic group to predict any potential failures.")

        st.write("**Q4: How does this system predict problems in wind turbines?**")
        st.write("Ans4: The system looks at historical data from the turbines and uses a machine learning model called XGBoost to find patterns. These patterns help predict if something might go wrong with a turbine.")

        st.write("**Q5: What is XGBoost, and why is it used for this prediction?**")
        st.write("A5: XGBoost is a type of machine learning model known for its accuracy and efficiency. It helps predict turbine failures by analyzing the data and learning from previous patterns to make reliable forecasts.")

        st.write("**Q6: How accurate is the system in predicting failures?**")
        st.write("Ans6: The system uses a measure called the **F1 score** to balance how well it predicts both failures and normal operation. The higher the F1 score, the better the system is at predicting failures.")

        st.write("**Q7: How does the system improve the accuracy of its predictions?**")
        st.write("Ans7: The system improves accuracy through a process called hyperparameter tuning. This involves adjusting the settings of the XGBoost model to find the best combination for making accurate predictions.")

        st.write("**Q8: What is hyperparameter tuning, and why is it important?**")
        st.write("Ans8: Hyperparameter tuning is the process of optimizing the settings of a machine learning model to improve its performance. In this system, it ensures that the XGBoost model performs at its best by finding the most effective settings.")

        st.write("**If you have any additional questions or need further clarification, feel free to ask in the chatbox. Your questions will be answered by llama3 , our advanced AI assistant.**")
        question = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            if question:
                answer = generate_answer(question)
                st.write(answer)
            else:
                st.write("Please enter a question.")

if __name__ == "__main__":
    main()
