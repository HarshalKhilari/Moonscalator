# Importing necessary libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

def search_symbols(search_str):
    # Input: search string
    # Output: No result found message or data frame of possible suggestions
    # Function to return data frame of suggestions appearing in yahoo finance website's search bar on entering a string

    start_time = time.time() # Tracking time when this function started running

    # Creating options object and setting preferences
    options=Options()
    options.set_preference("permissions.default.image", 2) # To limit image loading
    options.set_preference("permissions.default.video", 2) # To limit videos loading
    options.set_preference("media.autoplay.default", 1) # To limit autoplay media loading
    options.set_preference("media.autoplay.enabled.user-gestures-needed", False) # To stop autoplay of media
    options.add_argument("-headless") # Running browser in headless mode
    
    driver = webdriver.Firefox(options=options) # Launching Firefox browser with preferred options
    
    url = "https://finance.yahoo.com/" # URL of yahoo finance website
    
    driver.get(url) # Launching URL

    # Find search box and enter search string
    driver.find_element(By.ID, "yfin-usr-qry").send_keys(search_str)

    # Creating an explicit wait object to wait for 5 seconds
    wait = WebDriverWait(driver, 5)

    # Try for presense of 'Symbols' tab in search results and quit if explicit wait runs out and exit function
    # This is for cases where the search string has no symbol results
    try:
        sym = wait.until(EC.presence_of_element_located((By.XPATH, "//h3[contains(text(), 'Symbols')]")))
    except:
        driver.quit() # Quit driver
        print(f"This operation took {time.time() - start_time} seconds") # Print amount of time since start of function
        return "No relevant stocks found." # Return message that no stocks were found
    
    # Getting symbol data results by traversing XPATH
    sym_list = sym.find_element(By.XPATH, "../following-sibling::*[1]")

    # Extracting symbol data into soup
    soup = BeautifulSoup(sym_list.get_attribute("innerHTML"), "html.parser")
    
    driver.quit() # Quit driver as we've acquired soup

    items = soup.find_all('li') # Find every list item in symbols suggestion list

    # Creating lists for symbol, company name, and exchange
    symbol_list = []
    company_name_list = []
    exchange_list = []

    # Extracting symbol, company name, and exchange 
    for item in items:
        symbol = item.find('div', class_=re.compile(r'modules_quoteSymbol.*')).text.strip() # Extract Symbol
        # If stock is not publicly listed, it shows the symbol as private, in which case we ignore it
        if symbol == 'PRIVATE':
            continue
        symbol_list.append(symbol) # Add symbol to symbol list
        company_name = item.find('div', class_=re.compile(r'modules_quoteCompanyName.*')).text.strip() # Extract company name
        company_name_list.append(company_name) # Add company name to company name list
        exchange_name = item.find('span', class_=re.compile(r'modules_quoteSpan.*')).text.strip().split(' - ')[1] # Extract exchange
        exchange_list.append(exchange_name) # Add exchange name to exchange name list

    # Create a dataframe for the symbols, company names, and exchanges
    possible_symbols = pd.DataFrame({'Symbol':symbol_list, 'Company':company_name_list, 'Exchange':exchange_list}, 
                                    index=range(1, len(symbol_list)+1))

    # If a company name is suggested by not listed (i.e. it is private), an empty data frame will be created
    # If an empty dataframe is created, we print the time taken by the function and return a message that no stocks were found
    if possible_symbols.shape[0] == 0:
        print(f"This operation took {time.time() - start_time} seconds")
        return "No relevant stocks found."
    # If dataframe is generated, print time taken by the function and return dataframe
    print(f"Getting these suggestions took {time.time() - start_time} seconds")
    return possible_symbols



# Streamlit code 

import streamlit as st

st.set_page_config(page_title="Finding Ticker Symbol", page_icon="🔍")

st.sidebar.header("Find Ticker Symbol")

st.title("What's my Ticker?!")
stock_name = st.text_input(label="Enter the stock name", placeholder = "Enter the stock name here...", key="stock_name", label_visibility = 'collapsed')
if stock_name:
    with st.spinner("Searching for symbols..."):
        suggestions = search_symbols(stock_name)
    st.write(suggestions)