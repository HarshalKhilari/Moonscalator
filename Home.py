import streamlit as st

st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="🌍",
)

st.write("# Welcome to Stock Price Forecaster")

st.sidebar.success("Do you know your stock's ticker symbol?")

st.markdown(
    """
    With this webapp, you can...
    1. Forecast the stock price for the next month using the ticker symbol
    2. Find the ticker symbol using the stock name
    
    **👈 Select from the sidebar** as per your need.
"""
)