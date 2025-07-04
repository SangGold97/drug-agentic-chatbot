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
    
    async def search_urls(self, query: str, 
                          suffix_domain: str = " vinmec nhathuoclongchau pharmacity"
                          ) -> List[str]:
        """Search for URLs using DuckDuckGo"""
        try:
            # Add delay to avoid rate limiting
            await asyncio.sleep(1)

            # Run DuckDuckGo search in executor to avoid blocking
            loop = asyncio.get_event_loop()
            def _search():
                with DDGS(timeout=5) as ddgs:
                    return list(ddgs.text(
                        query + suffix_domain,
                        max_results=5,
                        region='wt-wt',
                        safesearch='moderate'
                    ))
            
            results = await loop.run_in_executor(None, _search)
            if not results:
                logger.warning(f"No results found for query: {query}")
                return []
            
            logger.info(f"Found {len(results)} URLs for query: {query}")
            
            # Filter by allowed domains and get unique URLs
            filtered_urls = []
            seen_urls = set()
            
            for result in results:
                url = result.get('href', '')
                if url and url not in seen_urls and self._is_allowed_domain(url):
                    filtered_urls.append(url)
                    seen_urls.add(url)
                    if len(filtered_urls) >= self.max_results:
                        break
            
            logger.info(f"Found {len(filtered_urls)} relevant URLs for query: {query}")
            return filtered_urls
        
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
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


# Test the WebSearchTool
if __name__ == "__main__":
    async def test_web_search_tool():
        start_time = asyncio.get_event_loop().time()
        
        logger.info("Initializing WebSearchTool...")
        web_search_tool = WebSearchTool()
        
        # Test search_and_fetch function with multiple queries
        logger.info("Testing search_and_fetch function...")
        
        test_queries = [
            "chỉ định và tác dụng phụ của paracetamol, bệnh đau đầu",
            # "tác dụng của paracetamol trong giảm đau?",
            # "có những loại thuốc giảm đau phổ biến nào?"
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
            logger.info(f"\nQuery: '{query}'")
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
        logger.info(f"\nTest Summary:")
        logger.info(f"- Total queries: {len(test_queries)}")
        logger.info(f"- Total web pages fetched: {total_contents}")
        logger.info(f"- Total execution time: {total_time:.2f}s")
        logger.info(f"- Average time per query: {total_time/len(test_queries):.2f}s")

    # Run the test
    asyncio.run(test_web_search_tool())

