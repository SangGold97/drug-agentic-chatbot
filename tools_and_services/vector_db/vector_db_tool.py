from database.milvus_manager import MilvusManager
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

class VectorDBTool:
    def __init__(self):
        self.milvus_manager = MilvusManager()
        self.top_k_knowledge = int(os.getenv('TOP_K_KNOWLEDGE', 10))
        self.top_k_intent = int(os.getenv('TOP_K_INTENT', 9))
    
    async def connect(self):
        """Connect to Milvus database"""
        try:
            # Connect to Milvus database
            await self.milvus_manager.connect()
            logger.info("Connected to Milvus database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus database: {e}")
            raise

    async def search(self, query_embedding: List[float],
                     collection_name: str) -> List[Dict]:
        """Search knowledge base or intent queries using vector similarity"""
        try:
            # Search vectors in specified collection
            results = await self.milvus_manager.search_vector(
                query_vector=query_embedding,
                collection_name=collection_name,
                top_k=self.top_k_knowledge if collection_name == "knowledge_base" else self.top_k_intent,
            )

            logger.info(f"VectorSearch successfully! Found {len(results)} similar chunks from {collection_name} collection for query")
            return results
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def insert(self, collection_name: str, documents: List[Dict[str, Any]]) -> Dict[str, str]:
        """Insert documents into specified collection"""
        try:
            # Insert documents into collection
            await self.milvus_manager.insert_documents(
                collection_name=collection_name,
                documents=documents
            )

            logger.info(f"Inserted {len(documents)} documents into {collection_name} collection")
            return {"status": "success", "message": f"Inserted {len(documents)} documents into {collection_name} collection"}
        
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return {"status": "error", "message": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        try:
            # Get collection statistics
            stats = self.milvus_manager.get_collection_stats()
            
            logger.info(f"Retrieved stats for {len(stats)} collections")
            return {
                "status": "success",
                "message": f"Retrieved stats for {len(stats)} collections",
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "status": "error", 
                "message": str(e),
                "stats": {}
            }

    def delete_collection(self, collection_name: str) -> Dict[str, str]:
        """Delete a collection"""
        try:
            # Delete collection from Milvus
            result = self.milvus_manager.delete_collection(collection_name)
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return {"status": "error", "message": str(e)}

    def health_check(self) -> Dict[str, str]:
        """Health check for vector search service"""
        try:
            return {"status": "healthy", "message": "Vector search service is ready"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}

