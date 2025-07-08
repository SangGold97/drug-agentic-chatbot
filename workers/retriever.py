import asyncio
import httpx
from typing import List, Dict, Optional, Tuple
from loguru import logger


class Retriever:
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize Retriever with base URL for the tools API
        
        Args:
            base_url: Base URL for the tools and services API
        """
        self.base_url = base_url
    
    async def _call_web_search(self, structured_query: str) -> Dict:
        """Call web search API and return results"""
        url = f"{self.base_url}/web_search/search_and_fetch"
        payload = {"structured_queries": [structured_query]}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result.get("results", {})
            else:
                logger.error(f"Web search failed with status {response.status_code}")
                return {}
    
    async def _call_vector_search(self, structured_query: str) -> List[Dict]:
        """Call vector search pipeline: embedding -> vector_db -> rerank"""
        # Step 1: Generate embedding
        embedding_url = f"{self.base_url}/embedding/generate_embedding"
        embedding_payload = {"texts": [structured_query]}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Generate embedding
            response = await client.post(embedding_url, json=embedding_payload)
            if response.status_code != 200:
                logger.error(f"Embedding generation failed with status {response.status_code}")
                return []
            
            embedding_result = response.json()
            query_embedding = embedding_result["embeddings"][0]
            
            # Step 2: Vector search
            vector_search_url = f"{self.base_url}/vector_db/search"
            vector_payload = {
                "query_embedding": query_embedding,
                "collection_name": "knowledge_base"
            }
            
            response = await client.post(vector_search_url, json=vector_payload)
            if response.status_code != 200:
                logger.error(f"Vector search failed with status {response.status_code}")
                return []
            
            vector_result = response.json()
            chunks = vector_result.get("results", [])
            
            # Step 3: Rerank results
            if not chunks:
                return []
            
            rerank_url = f"{self.base_url}/rerank/rerank"
            rerank_payload = {
                "query": structured_query,
                "chunks": chunks
            }
            
            response = await client.post(rerank_url, json=rerank_payload)
            if response.status_code != 200:
                logger.error(f"Reranking failed with status {response.status_code}")
                return chunks  # Return original chunks if reranking fails
            
            rerank_result = response.json()
            return rerank_result.get("reranked_chunks", [])
    
    async def run(self, structured_query: str, 
                  web_search: bool = True, vector_search: bool = True) -> Dict:
        """
        Run retrieval with specified search methods
        
        Args:
            structured_query: The query text to search with
            web_search: Whether to perform web search
            vector_search: Whether to perform vector search
            
        Returns:
            Dictionary containing search results
        """
        results = {}
        
        # Create tasks for concurrent execution
        tasks = []
        
        if web_search:
            tasks.append(self._call_web_search(structured_query))
        
        if vector_search:
            tasks.append(self._call_vector_search(structured_query))
        
        if not tasks:
            logger.warning("No search method specified")
            return results
        
        # Execute tasks concurrently if both are True
        if web_search and vector_search:
            web_results, vector_results = await asyncio.gather(*tasks)
            results["web_search"] = web_results
            results["vector_search"] = vector_results
        elif web_search:
            web_results = await tasks[0]
            results["web_search"] = web_results
        elif vector_search:
            vector_results = await tasks[0]
            results["vector_search"] = vector_results
        
        return results


async def main():
    """Test function for the Retriever class"""
    # Initialize retriever
    retriever = Retriever()
    
    # Test query
    test_query = "tác dụng của thuốc ibuprofen"

    logger.info(f"Testing Retriever with query: '{test_query}'")
    
    # Test 1: Both web search and vector search
    logger.info("Test 1: Both web search and vector search")
    results_both = await retriever.run(test_query, web_search=True, vector_search=True)
    logger.info(f"Results (both): {len(results_both)} result types")
    logger.info(f"Web search results: {results_both.get('web_search')}")
    logger.info(f"Vector search results: {results_both.get('vector_search', [])}")

    # Test 2: Only web search
    logger.info("Test 2: Only web search")
    results_web = await retriever.run(test_query, web_search=True, vector_search=False)
    logger.info(f"Results (web only): {len(results_web)} result types")
    logger.info(f"Web search results: {results_web.get('web_search')}")

    # Test 3: Only vector search
    logger.info("Test 3: Only vector search")
    results_vector = await retriever.run(test_query, web_search=False, vector_search=True)
    logger.info(f"Results (vector only): {len(results_vector)} result types")
    logger.info(f"Vector search results: {results_vector.get('vector_search')}")

    # Test 4: No search (edge case)
    logger.info("Test 4: No search methods")
    results_none = await retriever.run(test_query, web_search=False, vector_search=False)
    logger.info(f"Results (none): {len(results_none)} result types")
    
    logger.info("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())