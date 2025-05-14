from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

options = Options()
# Set a real browser user-agent
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
# Optional: run in non-headless mode
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)
query = "laptop"
file = 0

# Create data folder
os.makedirs("data", exist_ok=True)

for i in range(1, 20):
    driver.get(f"https://www.amazon.in/s?k={query}&page={i}&crid=4H22E92RYJXX&sprefix=laptop%2Caps%2C262&ref=nb_sb_noss_2")

    try:
        # Wait until product containers appear
        elems = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "puis-card-container"))
        )
    except:
        elems = []

    print(f"{len(elems)} items found")
    for elem in elems:
        print(elem)
        d = elem.get_attribute("outerHTML")
        with open(f"data/{query}_{file}.html", "w", encoding="utf-8") as f:
            f.write(d)
        file += 1

    time.sleep(3)

driver.quit()
