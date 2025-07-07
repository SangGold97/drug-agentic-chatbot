from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Set
import os
from dotenv import load_dotenv
from loguru import logger
import asyncio
import aiohttp
import re
from readability import Document
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random

load_dotenv()

class WebSearchTool:
    def __init__(self):
        self.max_results = int(os.getenv('MAX_SEARCH_RESULTS', 3))
        self.allowed_domains = [domain.strip() for domain in os.getenv('ALLOWED_DOMAINS', '').split(',')]
    
    def _is_allowed_domain(self, url: str) -> bool:
        """Check if URL is from allowed domain"""
        try:
            domain = urlparse(url).netloc.lower()
            return any(allowed in domain for allowed in self.allowed_domains)
        except:
            return False
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract main content from HTML using readability"""
        try:
            # Use readability to extract main content
            doc = Document(html_content)
            main_content_html = doc.summary()
            
            # Parse with BeautifulSoup for final text extraction
            soup = BeautifulSoup(main_content_html, 'html.parser')
            
            # Remove any remaining unwanted elements
            for element in soup(["script", "style", "nav", "footer", "aside"]):
                element.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
            text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines
            return text.strip()
            
        except Exception as e:
            logger.info(f"Cannot extract text with readability: {e}")
            # Fallback to basic extraction
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                return text
                
            except Exception as fallback_error:
                logger.error(f"Fallback text extraction also failed: {fallback_error}")
                return ""
    
    async def search_urls(self, query: str, max_retries = 2,
                          suffix_domain: str = " vinmec nhathuoclongchau pharmacity"
                          ) -> List[str]:
        """Search for URLs using DuckDuckGo with Selenium to avoid rate limiting."""
        try:
            # Random user agents to avoid detection
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"
            ]
            
            # Configure Chrome options to avoid detection
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument(f"--user-agent={random.choice(user_agents)}")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")

            # Initialize WebDriver
            driver = webdriver.Chrome(options=chrome_options)
            
            # Execute script to hide webdriver property
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            urls = []
            
            for attempt in range(max_retries):
                try:
                    # Random delay before each attempt
                    if attempt > 0:
                        delay = random.uniform(0.5, 1)
                        await asyncio.sleep(delay)
                    
                    # Open DuckDuckGo
                    driver.get("https://duckduckgo.com/")
                    
                    # Random short delay to simulate human behavior
                    await asyncio.sleep(random.uniform(0.1, 0.5))

                    # Wait for page to load and find the search box
                    wait = WebDriverWait(driver, 5)
                    search_box = wait.until(EC.presence_of_element_located((By.NAME, "q")))
                    
                    # Simulate human typing with random delays
                    search_text = f"{query}{suffix_domain}"
                    search_box.clear()
                    for char in search_text:
                        search_box.send_keys(char)
                        if random.random() < 0.1:  # 10% chance of pause
                            await asyncio.sleep(random.uniform(0.05, 0.1))
                    
                    # Random delay before pressing enter
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                    search_box.send_keys(Keys.RETURN)

                    # Wait for search results to load
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                    
                    # Try different selectors for search results
                    result_selectors = [
                        'a[data-testid="result-title-a"]',
                        'h2 a',
                        '.result__a',
                        'a.result__a',
                        '[data-testid="result-extras-url-link"]'
                    ]
                    
                    results = []
                    for selector in result_selectors:
                        results = driver.find_elements(By.CSS_SELECTOR, selector)
                        if results:
                            break

                    # Extract URLs from results
                    for result in results:
                        href = result.get_attribute('href')
                        if href:
                            urls.append(href)
                    
                    # If we found URLs, break out of retry loop
                    if urls:
                        break
                        
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise e

            # Ensure the driver is closed
            driver.quit()

            if not urls:
                return []

            # Filter by allowed domains and get unique URLs
            filtered_urls = []
            seen_urls = set()

            for url in urls:
                if url and url not in seen_urls and self._is_allowed_domain(url):
                    filtered_urls.append(url)
                    seen_urls.add(url)
                    if len(filtered_urls) >= self.max_results:
                        break
            
            logger.info(f"Found {len(filtered_urls)} relevant URLs for query: {query}")
            return filtered_urls

        except Exception as e:
            logger.error(f"An error occurred during web search with Selenium: {e}")
            return []

    async def _fetch_url_content(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Fetch content from URL asynchronously """
        try:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    html_content = await response.text()
                    text_content = self._extract_text_from_html(html_content)
                    return {
                        'url': url,
                        'content': text_content,
                        'success': True
                    }
                else:
                    return {'url': url, 'content': '', 'success': False}
                
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return {'url': url, 'content': '', 'success': False}
    
    async def fetch_web_content(self, urls: List[str]) -> List[Dict]:
        """Fetch content from multiple URLs asynchronously"""
        if not urls:
            return []
        
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [self._fetch_url_content(session, url) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter successful results
                successful_results = []
                for result in results:
                    if isinstance(result, dict) and result.get('success') and result.get('content'):
                        successful_results.append(result)
                
                return successful_results
        
        except Exception as e:
            logger.error(f"Failed to fetch web content: {e}")
            return []
    
    async def search_and_fetch(self, structured_queries: List[str]) -> Dict[str, List[Dict]]:
        """Search and fetch content for multiple structured queries"""
        if not structured_queries:
            return {}
        
        all_results = {}
        all_urls = set()
        
        # Search URLs for each query asynchronously
        search_tasks = [self.search_urls(aug_query) for aug_query in structured_queries]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process search results
        for query, urls in zip(structured_queries, search_results):
            if urls:
                all_results[query] = urls
                all_urls.update(urls)
            else:
                logger.warning(f"No URLs found for query: {query}")
                all_results[query] = []
        
        # Fetch content for all unique URLs
        if all_urls:
            web_contents = await self.fetch_web_content(list(all_urls))
            
            # Create URL to content mapping
            url_content_map = {item['url']: item['content'] for item in web_contents}
            
            # Map content back to queries
            query_contents = {}
            for query, urls in all_results.items():
                contents = []
                for url in urls:
                    if url in url_content_map:
                        contents.append({
                            'url': url,
                            'content': url_content_map[url]
                        })
                query_contents[query] = contents

            return query_contents

        return {query: [] for query in structured_queries}

    def health_check(self) -> Dict[str, str]:
        """Health check for web search service"""
        try:
            return {"status": "healthy", "message": "Web search service is ready"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}


# Test the WebSearchTool
if __name__ == "__main__":
    async def test_web_search_tool():
        start_time = asyncio.get_event_loop().time()
        
        logger.info("Initializing WebSearchTool...")
        web_search_tool = WebSearchTool()
        
        # Test search_and_fetch function with multiple queries
        logger.info("Testing search_and_fetch function...")
        
        test_queries = [
            # "chỉ định và tác dụng phụ của meloxicam",
            # "tác dụng của paracetamol trong giảm đau",
            "cách điều trị viêm gan B",
        ]
        
        logger.info(f"Searching and fetching content for {len(test_queries)} queries:")
        for i, query in enumerate(test_queries, 1):
            logger.info(f"  {i}. {query}")
        
        search_start = asyncio.get_event_loop().time()
        results = await web_search_tool.search_and_fetch(test_queries)
        search_time = asyncio.get_event_loop().time() - search_start
        
        logger.info(f"Search and fetch completed in {search_time:.2f}s")
        
        # Print detailed results
        total_contents = 0
        for query, contents in results.items():
            logger.info(f"Query: '{query}'")
            logger.info(f"Found {len(contents)} relevant web pages:")
            
            for i, content in enumerate(contents, 1):
                url = content['url']
                text_length = len(content['content'])
                snippet = content['content'][:1000] + '...'
                logger.info(f"  {i}. {url}")
                logger.info(f"     Length: {text_length} chars")
                logger.info(f"     Preview: {snippet}")
                total_contents += 1
        
        total_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Test Summary:")
        logger.info(f"- Total queries: {len(test_queries)}")
        logger.info(f"- Total web pages fetched: {total_contents}")
        logger.info(f"- Total execution time: {total_time:.2f}s")

    # Run the test
    asyncio.run(test_web_search_tool())

