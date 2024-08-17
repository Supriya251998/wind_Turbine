import streamlit as st
from sample_pages import show_sample_page_1, show_sample_page_2

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
    st.write("Welcome to the Introduction page about our model.")
    st.write("This model is designed to revolutionize how we approach data analysis and decision-making. Utilizing cutting-edge algorithms and advanced machine learning techniques, it provides deep insights and predictions with remarkable accuracy. Our model leverages large datasets and complex feature engineering to uncover patterns that are not immediately apparent. It is equipped with a user-friendly interface, making it accessible to both technical and non-technical users. Whether you are looking to optimize business processes, enhance customer experiences, or gain a competitive edge, this model offers a comprehensive solution tailored to meet your needs.")

    # Sub-menu for Introduction
    intro_page = st.selectbox("Select a section", ["Overview", "Details", "FAQs"])

    if intro_page == "Overview":
        st.subheader("Overview")
        st.write("Here is an overview of the model.")
        st.write("Our model integrates state-of-the-art machine learning techniques to deliver accurate and actionable insights. It features a robust architecture that supports a wide range of applications, from predictive analytics to data visualization. The model's scalability ensures that it can handle large volumes of data efficiently, making it suitable for various industries and use cases.")
        
        st.write("For more details on the methodologies used, you can refer to the following papers and resources:")
        st.markdown("- [Paper 1: Title of the Paper](https://example.com/paper1)")
        st.markdown("- [Paper 2: Title of the Paper](https://example.com/paper2)")
        st.markdown("- [Resource: Title of the Resource](https://example.com/resource)")
        
        st.write("### Key Features:")
        st.write("- **Scalability**: Efficiently handles large datasets.")
        st.write("- **Flexibility**: Adaptable to various data types and applications.")
        st.write("- **Accuracy**: Provides precise and reliable predictions.")
        st.write("- **User-Friendly Interface**: Easy to use for both technical and non-technical users.")

    elif intro_page == "Details":
        st.subheader("Details")
        st.write("Here are the details about the model.")
        st.write("The model uses advanced algorithms such as neural networks, ensemble methods, and natural language processing to analyze data. It includes components for data preprocessing, feature selection, and model evaluation. Each component is optimized for performance, ensuring that the model provides reliable and timely results. Additionally, the model offers customizable parameters, allowing users to fine-tune its performance based on specific needs.")

    elif intro_page == "FAQs":
        st.subheader("FAQs")
        st.write("Frequently Asked Questions about the model.")

        st.write("**Q1: What type of data does the model require?**")
        st.write("A1: The model can handle various types of data, including numerical, categorical, and textual data. It is designed to preprocess and normalize data to ensure consistent performance.")

        st.write("**Q2: How accurate is the model?**")
        st.write("A2: The model achieves high accuracy rates, with performance metrics varying depending on the specific use case and dataset. We recommend evaluating the model with your own data to assess its accuracy for your application.")

        st.write("**Q3: Can the model be integrated with existing systems?**")
        st.write("A3: Yes, the model is designed to be easily integrated with existing systems via APIs and other interfaces. It can be customized to fit into your current workflow and data infrastructure.")

        st.write("**Q4: What are the hardware and software requirements?**")
        st.write("A4: The model requires a standard computing environment with adequate processing power and memory. Specific requirements may vary based on the scale of data and complexity of the analysis. Detailed specifications are provided in the documentation.")

        st.write("**Q5: Is technical support available?**")
        st.write("A5: Yes, we offer comprehensive technical support, including documentation, tutorials, and customer service to assist with any issues or questions.")


if __name__ == "__main__":
    main()
