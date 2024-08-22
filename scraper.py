from selenium import webdriver
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
import re
import logging
import os
import time
import random
import pandas as pd
import numpy as np
from config import SCRAPER_TIMEOUT, CHROME_DRIVER_PATH, SCRAPER_MAX_RETRIES


USER_AGENTS = us_ag = pd.read_csv("https://gist.githubusercontent.com/pzb/b4b6f57144aea7827ae4/raw/cf847b76a142955b1410c8bcef3aabe221a63db1/user-agents.txt", sep="\t", header=None)
USER_AGENTS = USER_AGENTS.iloc[:, 0].copy()


def get_text(url, n_words=15):
    try:
        driver = None
        logging.warning(f"Initiated Scraping {url}")
        # user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ua = USER_AGENTS[np.random.randint(low=0, high=len(USER_AGENTS), size=1)]
        ua = ua.reset_index(drop=True)
        ua = ua[0]
        user_agent = ua
        options = uc.ChromeOptions() 
        options.add_argument("--headless") 
        options.add_argument(f"user-agent={user_agent}")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--disable-images")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        
        # options.add_argument("--disable-extensions")
        # options.add_argument("--autoplay-policy=no-user-gesture-required")
        # options.add_argument("--disable-infobars")
        # options.add_argument("--disable-gpu")

        driver = uc.Chrome(version_main=127, options=options, driver_executable_path=CHROME_DRIVER_PATH)
        time.sleep(random.uniform(0.5, 1.5))
        driver.set_page_load_timeout(SCRAPER_TIMEOUT)
        driver.set_script_timeout(SCRAPER_TIMEOUT)
        driver.implicitly_wait(3)     
        driver.get(url)
        elem = driver.find_element(By.TAG_NAME, "body").text
        sents = elem.split("\n")
        sentence_list = []
        for sent in sents:
            sent = sent.strip()
            if (len(sent.split()) >= n_words) and (len(re.findall(r"^\w.+[^\w\)\s]$", sent))>0):
                sentence_list.append(sent)
        driver.close()
        driver.quit()
        logging.warning("Closed Webdriver")
        logging.warning("Successfully scraped text")
        if len(sentence_list) < 3:
            raise Exception("Found nothing to scrape.")
        return "\n".join(sentence_list), ""
    except Exception as e: 
        logging.warning(str(e))
        if driver:
            driver.close()
            driver.quit()
            logging.warning("Closed Webdriver")
        err_msg = str(e).split('\n')[0]
        return "", err_msg


def scrape_text(url, n_words=15,max_retries=SCRAPER_MAX_RETRIES):
    scraped_text = ""
    scrape_error = ""
    try:
        n_tries = 1
        while (n_tries <= max_retries) and (scraped_text == ""):
            scraped_text, scrape_error = get_text(url=url, n_words=n_words)
            n_tries += 1
        return scraped_text, scrape_error
    except Exception as e:
        err_msg = str(e).split('\n')[0]
        return "", err_msg
