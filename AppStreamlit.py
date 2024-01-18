import streamlit as st
from pycaret.regression import load_model, predict_model 
import pandas as pd 

modelo = load_model("modelito2")
st.title("Inferencia en el precio de las casas  ;()")

OverallQual = st.number_input("Calidad de la casa")
YearBuilt = st.number_input("Año de construcción")
YearRemodAdd = st.number_input("Año de remodelación")
TotalBsmtSF = st.number_input("Tamaño sótano")
GarageCars = st.number_input("Nº de coches en garaje")
GrLivArea = st.number_input("Superficie habitable")
input_1stFlrSF= st.number_input("Área de la primera planta")
FullBath = st.number_input("Baños completos")
TotRmsAbvGrd = st.number_input("Nº de habitaciones")


if st.button("predecir"):
    input_data = pd.DataFrame([[OverallQual, YearBuilt, YearRemodAdd, TotalBsmtSF, GarageCars, GrLivArea, input_1stFlrSF, FullBath, TotRmsAbvGrd]], columns=['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GarageCars', 'GrLivArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd'])
    prediction = predict_model(modelo, data = input_data)
    prediction.rename(columns={'prediction_label': 'SalePrice'}, inplace=True)
    st.write(prediction)