import streamlit as st
import numpy as np
from PIL import Image
import pickle
from modules.funciones_ml import read_data_modelo, load_model_aws, read_data

import sklearn 
import pyarrow.parquet as pq
import pandas as pd
import plotly.express as px
import joblib
import dill


def ml():
    st.subheader(body = "Predicción del Precio :robot_face:")

    st.markdown(body = """En esta sección podrás ingresar las cararcterísticas de tu vehículo deseado y automáticamente predecimos su precio**_.""")

    st.sidebar.markdown("*"*10)
       
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    
    df = read_data()

    # with st.form("User's Input:"):


    coche_marca = st.selectbox(label = "Marca", options = df["Marca"].unique())


    df1 = df[df["Marca"] == coche_marca]

    coche_modelo = st.selectbox(label   = "Modelo",
                                    options = df1["Modelo"].unique())
        
    df2 = df1[df1["Modelo"] == coche_modelo]


    coche_combustible = st.selectbox(label = "Combustible", options = df2["Combustible"].unique())
    

    

    
 

    col5,col6, col7= st.columns(3)   
    
    with col5:

        coche_cv = st.number_input(label = "Cv",min_value=1, max_value=999 ) 

    with col6:

        coche_año = st.number_input(label = "Año", min_value=1978, max_value=2023 ) 

    with col7:
        coche_km =  st.number_input(label = "Kms", min_value=1, max_value=999999) 

    submitted = st.button("Submit")

    if submitted:
 

        yhat, r2 = load_model_aws(Marca = coche_marca, Modelo = coche_modelo, Combustible = coche_combustible,Cv = coche_cv, Año = coche_año, Kms = coche_km)
        

        column_name = str(yhat[0][0])[:6]

        st.write(f"Marca : {coche_marca}, Modelo : {coche_modelo}, Combustible : {coche_combustible}, "
         f", Cv : {coche_cv}, Año : {coche_año}, Kms : {coche_km}, Precio : {column_name} ")

        st.write(f'R-squared:{r2[0]}')






if __name__=="__ml__":
    ml()