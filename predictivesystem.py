# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle


import warnings
warnings.filterwarnings('ignore')
loaded_model=pickle.load(open('train3_model.sav','rb'))
input_data = (7.9,0.18,0.37,1.2,0.04,16,75,0.992,3.18,0.63,10.8)

# Changing the input data to numpy array
input_data_asnumpyarray=np.asarray(input_data)

## Reshape the data as we are predicting label for only one instance
input_data_reshaped = input_data_asnumpyarray.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0])==1:
    print('This is a good quality wine😊')
else:
    print('This is a poor quality wine😒')