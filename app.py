import streamlit as st 
import pandas as pd
import numpy as np
from joblib import load

st.set_page_config(page_title="Placement Prediction", page_icon=":coin:", layout="wide")
st.title('Placement Prediction App')
st.markdown("""
This app predicts whether a student will be placed based on their academic and personal details.
It uses a logistic regression model trained on historical placement data.
""")
st.sidebar.header('User Input Features')


model= load('./logistic_pipe_model.joblib')

gender = st.selectbox('Select Gender', ['F', 'M'])

tenth_percentage = st.number_input('Enter 10th Percentage', min_value=0.0, max_value=100.0, value=74.0)

twelfth_stream = st.selectbox('Select twelfth Stream', ['Commerce', 'Science', 'Arts'])

twelfth_percentage=st.number_input('Enter 12th Percentage', min_value=0.0, max_value=100.0, value=66.0)

graduation_percentage=st.number_input('Enter Graduation Percentage', min_value=0.0, max_value=100.0, value=66.0)

graduation_stream=st.selectbox('Select Graduation Stream',['Comm&Mgmt', 'Sci&Tech', 'Others'])

specialization=st.selectbox('Select Specialization', ['Mkt&HR', 'Mkt&Fin'])

mba_percentage=st.number_input('Enter MBA Percentage', min_value=0.0, max_value=100.0, value=60.23)

workex=st.selectbox('Work Experience', ['No', 'Yes'])


predict=st.button('Predict Placement')
if predict:
    predicted=model.predict(np.array([
        gender, np.float64(tenth_percentage), 
        np.float64(twelfth_percentage), twelfth_stream, 
        np.float64(graduation_percentage), graduation_stream ,
        workex , specialization ,np.float64(mba_percentage),
    ]).reshape(1, -1))
    
    if predicted[0] == 'Placed':
        st.success(predicted[0])
    else:
        st.error(predicted[0])