import pickle
import streamlit as st
import pandas as pd
import numpy as np


with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def loading_data(df):
    cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    user_inputs = {}
    for col in cols:
        min_value = round(df[col].min())
        max_value = round(df[col].max())
        user_inputs[col] = st.number_input(f"Enter value for {col} between {min_value} and {max_value}", key=col)
        
    if st.button("Predict"):
        X_test = pd.DataFrame([user_inputs])
        
        pred = loaded_model.predict(X_test)
        
        if pred == 1:
            st.success('Water is safe to drink', icon="✅")
        else:
            st.warning('WATER IS NOT SAFE TO DRINK', icon="⚠️")
    else:
        st.write("Thinking...")