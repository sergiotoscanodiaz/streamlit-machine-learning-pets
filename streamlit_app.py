import streamlit as st
import pandas as pd
import joblib

st.header("Streamlit Machine Learning App")

# Add a text input
height = st.number_input("Enter Height:")
weight = st.number_input("Enter Weight:")
eyes = st.selectbox("Select Eye Colour:", ("Blue","Brown"))

# Display the entered name
if st.button("Submit"):
    pet_model = joblib.load("pet_model.pkl")    
    X = pd.DataFrame([[height, weight, eyes]],
                     columns = ["Height", "Weight", "Eye"])
    X = X.replace(["Brown", "Blue"], [1, 0])
    prediction = pet_model.predict(X)[0]
    st.text(f"This instance is a {prediction}")