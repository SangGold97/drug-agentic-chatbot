from typing import Dict, List
import httpx
import json
import asyncio
from loguru import logger
import re

class StructuredQueryGenerator:
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize StructuredQueryGenerator with base URL for the tools API

        Args:
            base_url: Base URL for the tools and services API
        """
        self.base_url = base_url
    
    def _parse_text_response(self, response_text: str) -> str:
        """
        Parse the response text to extract structured_query
        
        Args:
            response_text: Raw response text from LLM service
            
        Returns:
            str: Extracted structured query
        """
        try:
            # Parse JSON
            response_data = json.loads(response_text)
            # response_content = response_data.get("response", "")
            structured_query = response_data.get("structured_query", "")
            return structured_query
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse response: {e}")
            # Try regex fallback
            match = re.search(r'"structured_query":\s*"([^"]*)"', response_text)
            if match:
                return match.group(1)
            return ""
        
    async def run(self, query: str) -> str:
        """
        Generate structured query from input query using LLM service
        
        Args:
            query: Input query string
            
        Returns:
            str: Structured query string
        """
        try:
            payload = {
                "service_name": "structured_query_generator",
                "query": query
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/llm/generate_response",
                    json=payload
                )
                response.raise_for_status()
                response_text = response.json().get("response", "")
                
                # Parse JSON response
                structured_query = self._parse_text_response(response_text)
                logger.info(f"Generated structured query: {structured_query}")
                
                return structured_query
                
        except Exception as e:
            logger.error(f"Error in structured query generation: {e}")
            return query  # Fallback to original query


async def main():
    """Test function for StructuredQueryGenerator"""
    generator = StructuredQueryGenerator()
    
    # Test queries
    test_queries = [
        "paracetamol chữa đau đầu có tác dụng phụ gì",
        "thuốc aspirin dùng cho người cao huyết áp",
        "tôi bị nấm da, phải uống thuốc gì?"
    ]
    
    for query in test_queries:
        print(f"\nInput query: {query}")
        try:
            structured_query = await generator.run(query)
            print(f"Structured query: {structured_query}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())