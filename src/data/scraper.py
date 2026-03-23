"""
AILS — Artificial Intelligence Learning System
Web Scraper Module
Handles static and dynamic web scraping for autonomous data acquisition.
Created by Cherry Computer Ltd.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
import time
import random


class AILSScraper:
    """
    AILS Static Web Scraper.
    Uses requests + BeautifulSoup for HTML scraping.

    Example:
        scraper = AILSScraper()
        data = scraper.scrape("https://example.com", tag="p", class_="content")
    """

    def __init__(self, headers: Optional[Dict] = None, timeout: int = 30,
                 rate_limit: float = 1.0):
        self.headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.logger = logging.getLogger("AILS.Scraper")

    def scrape(self, url: str, tag: str = "p",
               class_: Optional[str] = None) -> List[str]:
        """
        Scrape text content from a URL.

        Args:
            url: Target URL to scrape.
            tag: HTML tag to extract (default: 'p').
            class_: Optional CSS class to filter by.

        Returns:
            List of extracted text strings.
        """
        try:
            time.sleep(self.rate_limit + random.uniform(0, 0.5))
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            elements = (
                soup.find_all(tag, class_=class_)
                if class_
                else soup.find_all(tag)
            )
            texts = [el.get_text(strip=True) for el in elements
                     if el.get_text(strip=True)]
            self.logger.info(f"✅ Scraped {len(texts)} items from {url}")
            return texts
        except requests.HTTPError as e:
            self.logger.error(f"HTTP error {e.response.status_code} for {url}")
            return []
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return []

    def scrape_table(self, url: str,
                     table_id: Optional[str] = None) -> List[Dict]:
        """
        Scrape tabular data (e.g., financial tables) from a URL.

        Args:
            url: Target URL containing a table.
            table_id: Optional HTML id of the target table.

        Returns:
            List of dicts where keys are column headers.
        """
        try:
            time.sleep(self.rate_limit)
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            table = (
                soup.find("table", id=table_id) if table_id
                else soup.find("table")
            )
            if not table:
                self.logger.warning("No table found on the page.")
                return []
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = []
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                if cells and len(cells) == len(headers):
                    rows.append(dict(zip(headers, cells)))
            self.logger.info(f"✅ Scraped {len(rows)} table rows from {url}")
            return rows
        except Exception as e:
            self.logger.error(f"Table scraping error for {url}: {e}")
            return []

    def scrape_links(self, url: str, domain_filter: Optional[str] = None) -> List[str]:
        """
        Extract all hyperlinks from a page.

        Args:
            url: Target URL.
            domain_filter: Optional domain to filter links by.

        Returns:
            List of URLs found on the page.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            links = [a.get("href") for a in soup.find_all("a", href=True)]
            if domain_filter:
                links = [l for l in links if domain_filter in l]
            return links
        except Exception as e:
            self.logger.error(f"Link scraping error: {e}")
            return []


class DynamicScraper:
    """
    AILS Dynamic Scraper — for JavaScript-rendered content.
    Uses Selenium WebDriver to wait for page load before scraping.

    Example:
        scraper = DynamicScraper()
        data = scraper.scrape_dynamic("https://example.com", "#data-table")
    """

    def __init__(self, headless: bool = True, wait_timeout: int = 15):
        try:
            from selenium import webdriver
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.chrome.options import Options

            options = Options()
            if headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            self.driver = webdriver.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, wait_timeout)
            self.logger = logging.getLogger("AILS.DynamicScraper")
        except ImportError:
            raise ImportError(
                "Selenium is required for DynamicScraper. "
                "Install with: pip install selenium"
            )

    def scrape_dynamic(self, url: str, wait_selector: str) -> List[Dict]:
        """
        Scrape a dynamically rendered page.

        Args:
            url: Target URL with JS-rendered content.
            wait_selector: CSS selector to wait for before scraping.

        Returns:
            List of dicts from the first table found.
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC

        try:
            self.driver.get(url)
            self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
            )
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            table = soup.find("table")
            if not table:
                return []
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = []
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                if cells:
                    rows.append(dict(zip(headers, cells)))
            self.logger.info(f"✅ Dynamic scrape: {len(rows)} rows from {url}")
            return rows
        except Exception as e:
            self.logger.error(f"Dynamic scraping error for {url}: {e}")
            return []

    def __del__(self):
        """Ensure WebDriver is properly closed."""
        try:
            self.driver.quit()
        except Exception:
            pass
