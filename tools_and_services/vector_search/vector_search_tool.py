from database.milvus_manager import MilvusManager
from typing import List, Dict
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

class VectorSearchTool:
    def __init__(self):
        self.milvus_manager = MilvusManager()
        self.top_k_vector = int(os.getenv('TOP_K_VECTOR', 10))
        self.top_k_intent = int(os.getenv('TOP_K_INTENT', 9))

    async def search_knowledge_base(self, query_embedding: List[float]) -> List[Dict]:
        """Search knowledge base using vector similarity"""
        try:
            # Connect to Milvus database
            await self.milvus_manager.connect()
            
            # Search vectors in knowledge base
            results = await self.milvus_manager.search_vectors(
                query_vector=query_embedding,
                top_k=self.top_k_vector
            )

            logger.info(f"Found {len(results)} similar chunks from knowledge base for query")
            return results
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def search_intent(self, query_embedding: List[float]) -> List[Dict]:
        """Search for query intent classification"""
        try:
            # Connect to Milvus database
            await self.milvus_manager.connect()

            # Search intent
            intents = await self.milvus_manager.search_intent(
                query_embedding,
                top_k=self.top_k_intent
            )

            logger.info(f"Found {len(intents)} intents for query")
            return intents

        except Exception as e:
            logger.error(f"Intent search failed: {e}")
            return []

    async def close(self):
        """Close connections"""
        if self.milvus_manager:
            await self.milvus_manager.close()
