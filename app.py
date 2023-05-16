import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import os
import io
import joblib

import warnings
warnings.filterwarnings('ignore')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://media.makeameme.org/created/you-are-overfitting.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main(): 
    st.text("Immobili")

    absolute_path = os.path.dirname(__file__)
    relative_path = "reg_test.pkl"
    full_path = os.path.join(absolute_path, relative_path)
    
    newmodel = joblib.load(full_path)

    file = st.file_uploader("Carica un file CSV o Excel", type=["csv", "xlsx"])

    if file is not None:


        if os.path.splitext(file.name)[1] ==".xlsx":
            df = pd.read_excel(file, engine="openpyxl")
        else:
            df = pd.read_csv(file)
        

        y_pred = newmodel.predict(df)
        df["price"]=y_pred
        st.dataframe(df)

        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer,sheet_name='Elenco_tot', index=False)
        writer.save()
        output.seek(0)
        st.download_button(
        label="Scarica file Excel",
        data=output,
        file_name='reg_test.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    
    if file is None:

        #inference
        st.text("Try the model")
        input1 = st.number_input("Insert crim value:",)
        input2 = st.number_input("Insert zn value:",)
        input3 = st.number_input("Insert indus value:",)
        input4 = st.number_input("Insert chas value:",)
        input5 = st.number_input("Insert nox value:",)
        input6 = st.number_input("Insert rm value:",)
        input7 = st.number_input("Insert age value:",)
        input8 = st.number_input("Insert dis value:",)
        input9 = st.number_input("Insert rad value:",)
        input10 = st.number_input("Insert tax value:",)
        input11 = st.number_input("Insert ptratio value:",)
        input12 = st.number_input("Insert b value:",)
        input13 = st.number_input("Insert istat value:",)
        prediction = newmodel.predict([[input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13]])
        prediction = prediction[0]
        st.write(f"Predicted: {round(prediction,1)}$")



if __name__ == '__main__':
    add_bg_from_url() 
    main()
    
# streamlit run app.py        