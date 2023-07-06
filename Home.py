import streamlit as st
from PIL import Image



st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="🌍"
)

st.write("# Welcome to Stock Price Forecaster")
image = Image.open('delphi.png')
st.image(image)

st.sidebar.success("Do you know your stock's ticker symbol?")

st.markdown(
    """
    With this webapp, you can...
    1. Forecast the stock price for the next month using the ticker symbol
    2. Find the ticker symbol using the stock name
    
    **👈 Select from the sidebar** as per your need.


    App created by :  
    -> **Mr. Harshal Manoj Khilari**  
    -> **Mr. Oliver Antony Priyan**   
"""
)