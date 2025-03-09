import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def load_model():
    with open("poly_reg.pkl",'rb') as file:
        polynomial_model = pickle.load(file)
    return polynomial_model

def Prediction(model,data):
    data_reshaped = np.array(data).reshape(1,-1)
    prediction = model.predict(data_reshaped)
    return prediction

def main():
    st.title("Ice cream selling Prediction using Polynomial Regression")
    st.write("Enter the temparature to predit the sale of icecream")

    temperature = st.number_input("Temparature",min_value=0.0, max_value=50.0,value=25.0)

    if st.button("Predict"):
        model = load_model()
        input_data = [[temperature]]
        prediction = Prediction(model,input_data)
    
        st.success(f"Preciated icecream sales at {temperature}°C: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()