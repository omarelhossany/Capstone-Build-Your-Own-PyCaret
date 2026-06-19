import streamlit as st
import pandas as pd
import numpy as np

df = None
st.title('Exploratory analysis')
file = st.file_uploader('choose (csv) file')
try:
    df = pd.read_csv(file)
except:
    st.write('please enter (csv) file')
else:
    st.write(df)
choice = st.radio('what chart do you want',['area chart','bar chart','line chart','scatter chart',''],index=4)
first_col = st.text_input('First')
second_col = st.text_input('second')
try:
    is_clicked = st.button('Done')
    if is_clicked:
        if choice == 'area chart':
            st.area_chart(x=first_col,y=second_col, data=df)
        elif choice == 'bar chart':
            st.bar_chart(x=first_col,y=second_col,data=df)
        elif choice == 'line chart':
            st.line_chart(x=first_col,y=second_col, data=df)
        elif choice == 'scatter chart':
            st.scatter_chart(x=first_col,y=second_col,data=df)
except:
    st.write('!! somthing went wrong !!')    
st.title('pre-processing')
st.subheader('missing values')

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from feature_engine.imputation import CategoricalImputer
mean_imputer = SimpleImputer(strategy='mean',missing_values=np.nan)
mode_imputer = SimpleImputer(strategy='most_frequent',missing_values=np.nan)
frequent_imputer = CategoricalImputer(fill_value="missing")

def applying_str(obj):
    for i in df.columns: 
        if df[i].dtype == 'object' or df[i].dtype == 'str':
            df[i] = obj.fit_transform(df[[i]]).flatten()

def applying_int(obj):
    for i in df.columns: 
        if df[i].dtype == 'object' or df[i].dtype == 'str':
            continue
        df[[i]]=obj.fit_transform(df[[i]].values.reshape(-1,1))
try:
    if df.isna().sum().sum() == 0:
        st.write('your data does not contain none values')
    else:
        st.write('None values:')
        st.write(df.isna().sum())
        Choice = st.radio('what do you want to fill numerical none values.(it is applyed after selecting)',['Mean','Mode',''],index=2)
            
        if Choice == 'Mean':
                x = [] 
                applying_int(mean_imputer)
                st.write('numerical data is filled with { mean } value successfully')
                for i in df.columns :
                    if df[i].dtype == 'str' or df[i].dtype == 'object':
                        continue
                    x.append([i, df[i].isna().sum()])
                x=pd.DataFrame(x)
                st.write(x)
        elif Choice == 'Mode':
            x = [] 
            applying_int(mode_imputer)
            st.write('numerical data is filled with { mode } value successfully')
            for i in df.columns :
                if df[i].dtype == 'str' or df[i].dtype == 'object':
                    continue
                
                x.append([i, df[i].isna().sum()])
            x=pd.DataFrame(x)
            st.write(x)
        
        Choice = st.radio('what do you want to fill categorical none values',['Most_frequent','CategoricalImputer',''],index=2)
        
        if Choice == 'Most_frequent':
                applying_str(mode_imputer)
                st.write('numerical data is filled with most frequent value successfully')
                st.write(df.isna().sum())
        elif Choice =='CategoricalImputer':
                df = frequent_imputer.fit_transform(df)
                st.write('numerical data is filled with CategoricalImputer class imputer value successfully')
                st.write(df.isna().sum())
except: pass
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

x = []
st.subheader('encoding')
try:
    st.write(df)
except:
    pass
choice = st.radio('categorical encoding options',['Label Encoding','One-Hot encoding',''],index=2)
try:
    if choice == 'Label Encoding':
        applying_str(le)
    elif choice=='One-Hot encoding':
        df = pd.get_dummies(df,drop_first=True)
        for i in df.columns :
            if df[i].dtype == 'bool':
                df[i]=df[i].replace({True:1,False:0})
    st.write(df)
except:
    st.write('!! something went wrong !!') 



st.subheader('Predict')
import matplotlib.pyplot as plt

try:
    try:
        choice = st.radio('Model',['regression','classification',''],index=2)
        x=st.text_input('Features (sparate column names with [,] without any spaces )')
        x = x.split(',')
        y = st.text_input('Target')
        x.append(y)
        data = df.loc[:,x]
        x.pop()
        st.write('this may take a while')
        
    except:
        pass
    
    if choice == 'regression':
        from pycaret.regression import *
        s = setup(data=data, target = y, session_id = 1)
        best = compare_models()
        st.write('the best model is ',best)
        st.write('pediction: ')
        preds = predict_model(best, data=data)
        preds=pd.DataFrame(preds)
        st.write(preds,'Done')
    elif choice == 'classification':
        from pycaret.classification import *
        s = setup(data=data, target = y, session_id = 2)
        best = compare_models()
        st.write('the best model is ',best)
        st.write('pediction: ')
        preds =predict_model(best, data=data, raw_score = True)
        preds=pd.DataFrame(preds)
        st.write(preds,'Done')
except:
    pass
