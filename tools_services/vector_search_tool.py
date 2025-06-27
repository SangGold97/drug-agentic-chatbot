from database.milvus_manager import MilvusManager
from .embedding_tool import EmbeddingTool
from typing import List, Dict
import os
from dotenv import load_dotenv
import logging

load_dotenv()

class VectorSearchTool:
    def __init__(self):
        self.milvus_manager = None
        self.embedding_tool = EmbeddingTool()
        self.top_k = int(os.getenv('TOP_K_RETRIEVAL', 10))
        self.logger = logging.getLogger(__name__)
    
    async def search_knowledge_base(self, query: str, top_k: int = None) -> List[Dict]:
        """Search knowledge base using vector similarity"""
        if top_k is None:
            top_k = self.top_k
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_tool.embed_text(query)
            
            # Initialize Milvus manager if needed
            if self.milvus_manager is None:
                self.milvus_manager = MilvusManager()
                await self.milvus_manager.connect()
            
            # Search vectors in knowledge base
            results = await self.milvus_manager.search_vectors(
                collection_name="knowledge_base",
                query_vector=query_embedding,
                top_k=top_k
            )
            
            self.logger.info(f"Found {len(results)} similar chunks for query")
            return results
        
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    async def search_intent(self, query: str) -> str:
        """Search for query intent classification"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_tool.embed_text(query)
            
            # Initialize Milvus manager if needed
            if self.milvus_manager is None:
                self.milvus_manager = MilvusManager()
                await self.milvus_manager.connect()
            
            # Search intent
            intent = await self.milvus_manager.search_intent(query_embedding)
            
            return intent or "medical"  # Default to medical intent
        
        except Exception as e:
            self.logger.error(f"Intent search failed: {e}")
            return "medical"  # Default fallback
    
    async def close(self):
        """Close connections"""
        if self.milvus_manager:
            await self.milvus_manager.close()
