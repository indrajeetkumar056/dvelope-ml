# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:39:48 2024

@author: indra
"""

import numpy as np
import pickle
import streamlit as st



loaded_model=pickle.load(open("D:/trained_model.sav",'rb'))

#creating a fucntion for predcition 

def diabetedpred(input_data):
    
        
        
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        
        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        
        prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)
        
        if (prediction[0] == 0):
          return 'The person is not diabetic'
        else:
          return 'The person is diabetic'
      
def main():
    #giving title
    st.title('Diabetes prediciton')
    Pregnancies=st.text_input('No of Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure Value')
    SkinThickness=st.text_input('SkinThickness')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI Value')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction')
    Age=st.text_input('Give Your Age')
    #code for prediciton 
    diagnosis= ''
    
    #creating button for prediction
    
    if st.button('Diabetes testing'):
        diagnosis=diabetedpred([Pregnancies,Glucose,BloodPressure, SkinThickness, Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
    
    
    
    
if __name__=='__main__':
     main()
        
    
    
    
