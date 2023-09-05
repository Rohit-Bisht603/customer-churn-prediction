import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('features', 'rb') as fg:
    feat = joblib.load(fg)
    X = pd.DataFrame(feat)
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('scaler', 'rb') as fb:
    sc = joblib.load(fb)
ad = ['Chicago', 'Houston', 'Los Angeles', 'Miami', 'New York']
def pred(age, sub, bill, gb, gend, loc):
    loc_ind = np.where(X.columns.split("_")[1] == loc)[0]
    gen = np.where(X.columns.split('_')[1] == gend)[0]
    h = np.zeros(len(X.columns))
    h[0] = age
    h[1] = sub
    h[2] = bill
    h[3] = gb
    if loc_ind >= 0:
        if gen >= 0:
            h[gen] = 1
            h[loc_ind] = 1
        h[loc_ind] = 1
    if gen >= 0:
        h[gen] = 1

    return clf.predict(sc.transform(h))

def show_predictpage():
    st.title('Welcome to Customer Churn Prediction')
    st.caption('**:blue[Predict customer churn]**')

    loc = st.selectbox(
        '**Enter a location**',
        ad)
    age = st.text_input('**Age of customer**', 'Enter a value')
    sub = st.number_input('**Subscription length**', min_value=0, max_value=24, step=1)
    bill = st.number_input('**Monthly bill**')
    gb = st.number_input('**Data usage in Gb**')
    gend = st.radio(
        "**Gender**",
        ('Male', 'Female'))

    if st.button('Predict Churn'):
        p = pred(age, sub, bill, gb, gend, loc)
        st.success(p[0])