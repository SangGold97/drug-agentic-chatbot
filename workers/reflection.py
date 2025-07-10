from typing import List, Dict, Tuple
import httpx
import json
import re
from dotenv import load_dotenv
from loguru import logger
import asyncio

load_dotenv()

class Reflection:
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize Retriever with base URL for the tools API
        
        Args:
            base_url: Base URL for the tools and services API
        """
        self.base_url = base_url

    def _parse_context(self, context: Dict) -> Tuple[str, str]:
        """
        Parse context from web search results
        
        Args:
            context: Dict containing search results with possible 'web_search' key
            
        Returns:
            Tuple of (structured_query, formatted_context_string)
        """
        if 'web_search' not in context:
            return "", "Không có thông tin"

        web_search_results = context['web_search']
        
        # Get the first query and its results
        if not web_search_results:
            return "", "Không có thông tin"

        # Extract the first query as structured_query
        structured_query = list(web_search_results.keys())[0]
        results = web_search_results[structured_query]
        
        # Format context string
        context_parts = []
        for idx, result in enumerate(results):
            url = result.get('url', 'Không có URL')
            content = result.get('content', 'Không có thông tin')
            context_parts.append(f"Kết quả tìm kiếm {idx+1}:\nURL: {url}\nNội dung: {content}")
        
        formatted_context = "\n\n".join(context_parts)
        # logger.info(f"Parsed content:\n{formatted_context}")
        return structured_query, formatted_context

    async def run(self, structured_query: str, context: Dict) -> Dict:
        """
        Run reflection process: parse context and generate response via LLM service
        
        Args:
            structured_query: The query string
            context: Dict containing search results (web_search and/or vector_search)
            
        Returns:
            Dict containing reflection analysis with keys:
            - sufficient: bool
            - follow_up_query: str
        """
        try:
            # Parse context to get formatted string
            parsed_query, formatted_context = self._parse_context(context)
            # logger.info(f"Parsed query: {parsed_query}")
            # logger.info(f"Formatted context: {formatted_context}")
            
            # Use provided structured_query or fall back to parsed one
            query_to_use = structured_query if structured_query else parsed_query
            logger.info(f"Structured query using for reflection: {query_to_use}")
            
            # Prepare request payload for LLM service
            payload = {
                "service_name": "reflection",
                "structured_query": query_to_use,
                "context": formatted_context
            }
            
            # Send request to LLM service
            url = f"{self.base_url}/llm/generate_response"
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                
                if response.status_code != 200:
                    logger.error(f"LLM service request failed with status {response.status_code}")
                    return {
                        "sufficient": False,
                        "follow_up_query": query_to_use
                    }
                
                llm_response = response.json()
                response_text = llm_response.get("response", "")
                
                # Parse LLM response using text parser since it returns escaped JSON
                return self._parse_text_response(response_text)
                    
        except Exception as e:
            logger.error(f"Error in reflection run: {e}")
            return {
                "sufficient": False,
                "follow_up_query": query_to_use
            }
    
    def _parse_text_response(self, response_text: str) -> Dict:
        """
        Parse text response to extract reflection components from JSON string
        
        Args:
            response_text: Raw text response from LLM (JSON format)
            
        Returns:
            Dict with parsed components
        """
        try:
            # Direct JSON parsing
            parsed_response = json.loads(response_text)
            
            return {
                "sufficient": parsed_response.get("sufficient"),
                "follow_up_query": parsed_response.get("follow_up_query", "")
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.warning("Failed to parse JSON response, using text parsing fallback")
            
            sufficient = '"sufficient": true' in response_text.lower()
            
            # Extract follow-up query
            follow_up_query = ""
            match = re.search(r'"follow_up_query":\s*"([^"]*)"', response_text, re.IGNORECASE)
            if match:
                follow_up_query = match.group(1)
            
            return {
                "sufficient": sufficient,
                "follow_up_query": follow_up_query
            }

async def main():
    """Test function for the Reflection class"""
    # Initialize reflection
    reflection = Reflection()
    
    # Test query
    test_query = "lưu ý khi sử dụng meloxicam, liều dùng của thuốc meloxicam"
    
    # Mock context with web search results
    test_context = {
        "web_search": {
            test_query: [{
                "url": "https://example.com/meloxicam",
                "content": "Meloxicam là một loại thuốc chống viêm không steroid (NSAID) được sử dụng để giảm đau và viêm."
            }]
        }
    }
    
    logger.info(f"Testing Reflection with query: '{test_query}'")
    
    # Test reflection
    result = await reflection.run(test_query, test_context)
    
    logger.info("Reflection result:")
    logger.info(f"  Sufficient: {result.get('sufficient')}")
    logger.info(f"  Follow-up query: {result.get('follow_up_query')}")
    
    logger.info("Test completed!")

if __name__ == "__main__":
    asyncio.run(main())


