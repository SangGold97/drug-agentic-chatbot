from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Set
import os
from dotenv import load_dotenv
import logging
import asyncio
import aiohttp

load_dotenv()

class WebSearchTool:
    def __init__(self):
        self.max_results = int(os.getenv('MAX_SEARCH_RESULTS', 3))
        self.allowed_domains = os.getenv('ALLOWED_DOMAINS', '').split(',')
        self.allowed_domains = [domain.strip() for domain in self.allowed_domains if domain.strip()]
        self.logger = logging.getLogger(__name__)
    
    def _is_allowed_domain(self, url: str) -> bool:
        """Check if URL is from allowed domain"""
        try:
            domain = urlparse(url).netloc.lower()
            return any(allowed in domain for allowed in self.allowed_domains)
        except:
            return False
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML"""
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
            
            return text[:5000]  # Limit to 5000 characters
        except Exception as e:
            self.logger.error(f"Failed to extract text from HTML: {e}")
            return ""
    
    async def _fetch_url_content(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Fetch content from URL asynchronously"""
        try:
            async with session.get(url, timeout=10) as response:
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
            self.logger.error(f"Failed to fetch {url}: {e}")
            return {'url': url, 'content': '', 'success': False}
    
    async def search_urls(self, query: str) -> List[str]:
        """Search for URLs using DuckDuckGo"""
        try:
            # Run DuckDuckGo search in executor to avoid blocking
            loop = asyncio.get_event_loop()
            def _search():
                with DDGS() as ddgs:
                    return list(ddgs.text(
                        query, 
                        max_results=self.max_results * 3,  # Get more to filter
                        region='wt-wt',
                        safesearch='moderate'
                    ))
            
            results = await loop.run_in_executor(None, _search)
            
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
            
            self.logger.info(f"Found {len(filtered_urls)} relevant URLs for query: {query}")
            return filtered_urls
        
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
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
            self.logger.error(f"Failed to fetch web content: {e}")
            return []
    
    async def search_and_fetch(self, aug_queries: List[str]) -> Dict[str, List[Dict]]:
        """Search and fetch content for multiple augmented queries"""
        if not aug_queries:
            return {}
        
        all_results = {}
        all_urls = set()
        
        # Search URLs for each query
        for aug_query in aug_queries:
            urls = await self.search_urls(aug_query)
            all_results[aug_query] = urls
            all_urls.update(urls)
        
        # Fetch content for all unique URLs
        if all_urls:
            web_contents = await self.fetch_web_content(list(all_urls))
            
            # Create URL to content mapping
            url_content_map = {item['url']: item['content'] for item in web_contents}
            
            # Map content back to queries
            query_contents = {}
            for aug_query, urls in all_results.items():
                contents = []
                for url in urls:
                    if url in url_content_map:
                        contents.append({
                            'url': url,
                            'content': url_content_map[url]
                        })
                query_contents[aug_query] = contents
            
            return query_contents
        
        return {query: [] for query in aug_queries}
    
    def merge_and_deduplicate_content(self, query_contents: Dict[str, List[Dict]]) -> str:
        """Merge content from all queries and remove duplicates"""
        all_content = []
        seen_urls = set()
        
        for aug_query, contents in query_contents.items():
            for item in contents:
                url = item['url']
                if url not in seen_urls:
                    all_content.append(item['content'])
                    seen_urls.add(url)
        
        # Combine all content
        merged_content = '\n\n'.join(all_content)
        
        # Limit total content size
        if len(merged_content) > 10000:
            merged_content = merged_content[:10000] + "... [Content truncated]"
        
        return merged_content
