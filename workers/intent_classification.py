from typing import Optional, Dict, Any, List
from loguru import logger
import httpx

class IntentClassification:
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize IntentClassification with base URL for the tools API
        
        Args:
            base_url: Base URL for the tools and services API
        """
        self.base_url = base_url

    async def _create_embedding(self, query: str) -> List[float]:
        """Generate embedding for query"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embedding/generate_embedding",
                json={"texts": [query]}
            )
            response.raise_for_status()
            return response.json()["embeddings"][0]

    async def _search_intent(self, embedding: List[float]) -> List[Dict[str, Any]]:
        """Search similar intents using vector database"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/vector_db/search",
                json={
                    "query_embedding": embedding,
                    "collection_name": "intent_queries"
                }
            )
            response.raise_for_status()
            return response.json()["results"]

    def _count_label(self, search_results: List[Dict[str, Any]]) -> str:
        """Count intent labels and return the most frequent one"""
        medical_count = sum(1 for result in search_results if result.get("intent_label") == "medical")
        general_count = sum(1 for result in search_results if result.get("intent_label") == "general")
        
        return "medical" if medical_count > general_count else "general"

    async def run(self, query: str) -> str:
        """
        Classify intent for a given query
        
        Args:
            query: The input query to classify
            
        Returns:
            Intent label: 'medical' or 'general'
        """
        try:
            # Step 1: Generate embedding for query
            embedding = await self._create_embedding(query)
            logger.info(f"Generated embedding for query: {query}")
            
            # Step 2: Search similar intents
            search_results = await self._search_intent(embedding)
            logger.info(f"Found {len(search_results)} similar intents")
            
            # Step 3: Count labels and return most frequent
            intent_label = self._count_label(search_results)
            logger.info(f"Classified intent as: {intent_label}")
            
            return intent_label
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return "general"  # Default fallback


async def main():
    """Test the IntentClassification functionality"""
    # Initialize intent classifier
    classifier = IntentClassification()
    
    # Test queries
    test_queries = [
        "chào bạn, hôm nay trời đẹp quá, tôi nên đi chơi ở đâu?",
        "tôi bị nấm da, phải uống thuốc gì?",
        "Tôi bị đau đầu, có thuốc nào giúp không?",
        "Tôi bị cảm cúm, nên làm gì?"
    ]
    
    print("Testing Intent Classification:")
    print("-" * 40)
    
    for query in test_queries:
        try:
            intent = await classifier.run(query)
            print(f"Query: {query}")
            print(f"Intent: {intent}")
            print("-" * 40)
        except Exception as e:
            print(f"Error testing query '{query}': {e}")
            print("-" * 40)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())



