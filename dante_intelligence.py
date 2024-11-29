import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import logging
import os
from dotenv import load_dotenv

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)

def initialize_driver():
    """Initialize and return Chrome WebDriver with options."""
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")
    # Add headless mode for server deployment
    chrome_options.add_argument("--headless")
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    return driver

def perform_login(driver, wait):
    """Handle the login process."""
    try:
        # Get credentials
        email = os.getenv('UTR_EMAIL')
        password = os.getenv('UTR_PASSWORD')
        
        if not email or not password:
            st.error("Credentials not found in environment variables.")
            return False
            
        login_url = "https://app.utrsports.net/login"
        driver.get(login_url)
        time.sleep(3)
        
        # Login sequence
        email_field = wait.until(EC.presence_of_element_located((By.ID, "emailInput")))
        email_field.send_keys(email)
        
        password_field = driver.find_element(By.ID, "passwordInput")
        password_field.send_keys(password)
        
        sign_in_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='SIGN IN']"))
        )
        driver.execute_script("arguments[0].click();", sign_in_button)
        
        # Handle Continue button
        continue_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Continue']"))
        )
        continue_button.click()
        
        return True
        
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False

def perform_search(driver, wait, search_term):
    """Execute search and navigate to player stats."""
    try:
        # Navigate to search page
        base_url = "https://app.utrsports.net"
        driver.get(base_url)
        time.sleep(5)
        
        # Find and interact with search
        search_wrapper = wait.until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR, 
                "div.nav-search-wrapper div.d-none.d-lg-block div.globalSearch__globalSearchWrapper__NglK2"
            ))
        )
        
        search_container = search_wrapper.find_element(
            By.CSS_SELECTOR,
            "div.globalSearch__globalSearchContainer__3_82H"
        )
        
        input_container = search_container.find_element(
            By.CSS_SELECTOR,
            "div.globalSearch__globalSearchInputContainer__35Wld"
        )
        
        search_input = input_container.find_element(
            By.CSS_SELECTOR,
            "input[data-testid='globalSearch-searchInputButton-eUX6nl19']"
        )
        
        # Click expander if needed
        if not search_input.is_displayed():
            expander = search_wrapper.find_element(
                By.CSS_SELECTOR,
                "div.globalSearch__searchExpander__2jpEM"
            )
            driver.execute_script("arguments[0].click();", expander)
            time.sleep(2)
        
        # Enter search term
        search_input.clear()
        search_input.send_keys(search_term)
        time.sleep(2)
        
        # Click "SEE ALL"
        see_all_link = wait.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//a[contains(., 'SEE ALL')]"
            ))
        )
        see_all_link.click()
        time.sleep(3)
        
        # Click first result
        first_result = wait.until(
            EC.element_to_be_clickable((
                By.CSS_SELECTOR,
                "div.search__cardContainer__1Z9Ee > a"
            ))
        )
        driver.execute_script("arguments[0].click();", first_result)
        time.sleep(3)
        
        # Navigate to Stats tab
        stats_button = wait.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[contains(@class, 'btn-tab')][.//div[contains(text(), 'Stats')]]"
            ))
        )
        driver.execute_script("arguments[0].click();", stats_button)
        time.sleep(3)
        
        return True
        
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return False

def scrape_stats_data(driver):
    """Scrape the stats data from the page."""
    try:
        # Add your data scraping logic here
        # This is a placeholder - modify based on the actual data you want to scrape
        data = {
            'Date': [],
            'Event': [],
            'Result': []
        }
        
        # Return the scraped data
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Data scraping failed: {str(e)}")
        return None

def main():
    st.title("UTR Stats Scraper")
    
    # Create sidebar elements
    with st.sidebar:
        st.header("Search Parameters")
        search_term = st.text_input("Player Name")
        search_button = st.button("Search")
    
    if search_button and search_term:
        with st.spinner("Processing..."):
            try:
                # Initialize driver
                driver = initialize_driver()
                wait = WebDriverWait(driver, 20)
                
                # Login process
                st.info("Logging in...")
                if not perform_login(driver, wait):
                    st.error("Login failed")
                    driver.quit()
                    return
                
                # Search process
                st.info(f"Searching for {search_term}...")
                if not perform_search(driver, wait, search_term):
                    st.error("Search failed")
                    driver.quit()
                    return
                
                # Scrape data
                st.info("Scraping stats data...")
                df = scrape_stats_data(driver)
                
                if df is not None:
                    # Display the results
                    st.success("Data retrieved successfully!")
                    st.dataframe(df)
                    
                    # Add download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name=f"{search_term}_stats.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            finally:
                # Clean up
                driver.quit()
    
    # Add instructions
    if not search_button:
        st.info("Enter a player name in the sidebar and click 'Search' to get their stats.")

if __name__ == "__main__":
    main()