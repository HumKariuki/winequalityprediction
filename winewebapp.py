# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:19:41 2024

@author: LENOVO
"""
import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open("C:/Users/LENOVO/OneDrive/Desktop/ML/train2_model.sav", 'rb'))

def wineprediction(input_data):
    # Changing the input data to numpy array
    input_data_asnumpyarray = np.asarray(input_data)

    ## Reshape the data as we are predicting label for only one instance
    input_data_reshaped = input_data_asnumpyarray.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    if (prediction[0] == 1):
        return 'This is a good quality wine😊'
    else:
        return 'This is a poor quality wine😒'

def main():
    # Giving a title
    st.title('Wine Prediction Web App')
    fixed_acidity = st.text_input('Fixed Acidity')
    volatile_acidity = st.text_input('Volatile Acidity')
    citric_acid = st.text_input('Citric Acid')
    residual_sugar = st.text_input('Residual Sugar')
    chlorides = st.text_input('Chlorides')
    free_sulfur_dioxide = st.text_input('Free Sulfur Dioxide')
    total_sulfur_dioxide = st.text_input('Total Sulfur Dioxide')
    density = st.text_input('Density')
    ph = st.text_input('pH')
    sulphates = st.text_input('Sulphates')
    alcohol = st.text_input('Alcohol')
   
    # Code for prediction
    wine = ''
   
    # Creating a button for prediction
    if st.button('Wine Result'):
        wine = wineprediction([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol])
       
    st.success(wine)
   
if __name__ == '__main__':
    main()
