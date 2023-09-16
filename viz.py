import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def hist_plot(raw_df):
    option = st.selectbox("Which Histogram do you want to see?", ("ph", "Trihalomethanes", "Sulfate"))
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=raw_df,x=option,stat='density')
    st.pyplot(fig)
    
def box_plot(raw_df):
    option = st.selectbox("Which Boxplot do you want to see?", ("ph", "Trihalomethanes", "Sulfate"))
    fig = plt.figure(figsize=(10, 4))
    sns.boxplot(data=raw_df,x=option)
    st.pyplot(fig)
    
def count_plot(raw_df):
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(data = raw_df, x = 'Potability')
    st.pyplot(fig)
    
def feature_importance(df):
    importances = loaded_model.feature_importances_
    X = df.drop(['Potability', 'Unnamed: 0'], axis = 1)
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]

    fig = plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    st.pyplot(fig)