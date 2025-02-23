import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np
from io import BytesIO
from fuzzywuzzy import process 

st.set_page_config(page_title="Data Sweeper", layout='wide')

st.sidebar.title("Settings")
uploaded_files = st.sidebar.file_uploader("Upload your files (CSV or Excel):", type=["csv", "xlsx"], accept_multiple_files=True)

def load_data(file):
    file_ext = os.path.splitext(file.name)[-1].lower()
    if file_ext == ".csv":
        return pd.read_csv(file)
    elif file_ext == ".xlsx":
        return pd.read_excel(file)
    else:
        return None

def clean_data(df):
    st.sidebar.subheader("🛠 Data Cleaning Options")
    
    # Handling Missing Values
    missing_option = st.sidebar.selectbox("Handle Missing Values:", ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Forward Fill", "Backward Fill", "Custom Value"])
    if missing_option == "Drop Rows":
        df.dropna(inplace=True)
    elif missing_option == "Fill with Mean":
        df.fillna(df.mean(), inplace=True)
    elif missing_option == "Fill with Median":
        df.fillna(df.median(), inplace=True)
    elif missing_option == "Fill with Mode":
        df.fillna(df.mode().iloc[0], inplace=True)
    elif missing_option == "Forward Fill":
        df.fillna(method='ffill', inplace=True)
    elif missing_option == "Backward Fill":
        df.fillna(method='bfill', inplace=True)
    elif missing_option == "Custom Value":
        custom_value = st.sidebar.text_input("Enter Custom Value:")
        df.fillna(custom_value, inplace=True)
    
    # Removing Outliers
    if st.sidebar.checkbox("Remove Outliers using IQR"): 
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Data Type Correction
    if st.sidebar.checkbox("Convert Columns to Numeric (if applicable)"):
        df = df.apply(pd.to_numeric, errors='coerce')
    if st.sidebar.checkbox("Convert Date Columns"):
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                continue
    
    # Standardizing Categorical Data
    if st.sidebar.checkbox("Standardize Categorical Data (Lowercase)"):
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.lower().str.strip()
    
    # String Cleaning
    if st.sidebar.checkbox("Remove Special Characters from Text Columns"):
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
    
    # Removing Duplicates
    if st.sidebar.checkbox("Remove Duplicates"):
        df.drop_duplicates(inplace=True)
    
    return df

dataframes = {}

if uploaded_files:
    for file in uploaded_files:
        df = load_data(file)
        if df is None:
            st.sidebar.error(f"Unsupported file type: {file.name}")
            continue
        
        df = clean_data(df)
        dataframes[file.name] = df
        
        st.write(f"### 🔍 Preview: {file.name}")
        st.dataframe(df.head())
        
        # Column Selection
        st.sidebar.subheader(f"Column Selection - {file.name}")
        selected_columns = st.sidebar.multiselect(f"Choose columns for {file.name}", df.columns, default=df.columns.tolist())
        df = df[selected_columns]
        
        # Data Visualization
        st.subheader(f"📊 Data Visualization - {file.name}")
        visualization_type = st.selectbox("Choose visualization type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Pie Chart", "Heatmap", "Box Plot"], key=file.name)
        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if visualization_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
            x_axis = st.selectbox("Select X-axis:", numeric_columns + categorical_columns, key=f"x_{file.name}")
            y_axis = st.selectbox("Select Y-axis:", numeric_columns, key=f"y_{file.name}")
            
            if visualization_type == "Bar Chart":
                fig = px.bar(df, x=x_axis, y=y_axis)
            elif visualization_type == "Line Chart":
                fig = px.line(df, x=x_axis, y=y_axis)
            elif visualization_type == "Scatter Plot":
                fig = px.scatter(df, x=x_axis, y=y_axis)
            
            st.plotly_chart(fig)
        
        elif visualization_type == "Histogram":
            hist_col = st.selectbox("Select Column:", numeric_columns, key=f"hist_{file.name}")
            fig = px.histogram(df, x=hist_col)
            st.plotly_chart(fig)
        
        elif visualization_type == "Pie Chart":
            category_col = st.selectbox("Select Category Column:", categorical_columns, key=f"pie_{file.name}")
            value_col = st.selectbox("Select Value Column:", numeric_columns, key=f"pie_val_{file.name}")
            fig = px.pie(df, names=category_col, values=value_col)
            st.plotly_chart(fig)
        
        # File Conversion
        st.sidebar.subheader(f"Export Options - {file.name}")
        conversion_type = st.sidebar.radio(f"Convert {file.name} to:", ["CSV", "Excel"], key=f"convert_{file.name}")
        if st.sidebar.button(f"Convert {file.name}"):
            buffer = BytesIO()
            if conversion_type == "CSV":
                df.to_csv(buffer, index=False)
                file_name = file.name.replace(os.path.splitext(file.name)[-1], ".csv")
                mime_type = "text/csv"
            else:
                df.to_excel(buffer, index=False, engine='xlsxwriter')
                file_name = file.name.replace(os.path.splitext(file.name)[-1], ".xlsx")
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            buffer.seek(0)
            
            st.sidebar.download_button(
                label=f"Download {file.name} as {conversion_type}",
                data=buffer,
                file_name=file_name,
                mime=mime_type
            )

st.success("🎆🥳🙌 All files processed successfully!")
