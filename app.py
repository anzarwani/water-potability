import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from viz import hist_plot, count_plot, box_plot, feature_importance
from ml import loading_data
from notebook import notebook_show

df = pd.read_csv("data.csv")
raw_df = pd.read_csv("raw_data.csv")
with st.sidebar:
    choice = st.radio(
        "What do you want to do?",
        ("Home", "Explore Data", "Do Prediction", "Show Notebook")
    )
if choice == 'Home':
    st.title("Water Potabilty Data Viz and Prediction")

    st.header("Overview")
    st.divider()
    st.write("Water potability refers to the suitability of water for human consumption without posing health risks. It depends on various factors, including chemical, physical, and biological characteristics of the water. Monitoring and ensuring water potability is vital to safeguard public health and prevent waterborne diseases. Stringent regulations and testing protocols are in place to assess and maintain the quality of drinking water, aiming to provide safe and clean water to communities around the world.")
    st.divider()
    image = Image.open('water.jpg')

    st.image(image, caption='water')
    st.write("Data Sourced from Kaggle [link](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability)")
    
elif choice == 'Explore Data':
    st.title("Data Analysis")
    st.divider()
    hist_plot(raw_df)
    st.divider()
    box_plot(raw_df)
    st.divider()
    st.subheader("Count Plot of Water Being potable or not")
    st.divider()
    count_plot(raw_df)
    st.divider()
    st.subheader("Feature Importance")
    feature_importance(df)
    
elif choice == "Do Prediction":
    st.title("Prediction")
    loading_data(df)
    
elif choice == "Show Notebook":
    st.title("Kaggle Notebook")
    st.divider()
    notebook_show()
    
    
