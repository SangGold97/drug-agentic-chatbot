from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging
import asyncio

load_dotenv()

class MilvusManager:
    def __init__(self):
        self.host = os.getenv('MILVUS_HOST', 'localhost')
        self.port = int(os.getenv('MILVUS_PORT', 19530))
        self.db_name = os.getenv('MILVUS_DB', 'drug_chatbot')
        self.vector_dim = int(os.getenv('VECTOR_DIMENSION', 768))
        self.connection_alias = "default"
        self.logger = logging.getLogger(__name__)
    
    async def connect(self):
        """Establish connection to Milvus"""
        try:
            # Run connection in thread pool since pymilvus is sync
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: connections.connect(
                    alias=self.connection_alias,
                    host=self.host,
                    port=self.port
                )
            )
            self.logger.info("Connected to Milvus database")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    async def create_knowledge_base_collection(self):
        """Create knowledge base collection"""
        collection_name = "knowledge_base"
        
        loop = asyncio.get_event_loop()
        
        # Check if collection exists
        exists = await loop.run_in_executor(None, utility.has_collection, collection_name)
        if exists:
            self.logger.info(f"Collection {collection_name} already exists")
            return Collection(collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        schema = CollectionSchema(fields, "Knowledge base collection")
        
        # Create collection in thread pool
        collection = await loop.run_in_executor(None, Collection, collection_name, schema)
        
        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        await loop.run_in_executor(None, collection.create_index, "vector", index_params)
        
        self.logger.info(f"Created collection {collection_name}")
        return collection
    
    async def create_intent_queries_collection(self):
        """Create intent queries collection"""
        collection_name = "intent_queries"
        
        loop = asyncio.get_event_loop()
        
        # Check if collection exists
        exists = await loop.run_in_executor(None, utility.has_collection, collection_name)
        if exists:
            self.logger.info(f"Collection {collection_name} already exists")
            return Collection(collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="intent_label", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        schema = CollectionSchema(fields, "Intent queries collection")
        collection = await loop.run_in_executor(None, Collection, collection_name, schema)
        
        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        await loop.run_in_executor(None, collection.create_index, "vector", index_params)
        
        self.logger.info(f"Created collection {collection_name}")
        return collection
    
    async def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        """Insert documents into collection"""
        try:
            collection = Collection(collection_name)
            
            # Prepare data for insertion
            data = []
            for doc in documents:
                data.append([
                    doc.get('content', ''),
                    doc.get('metadata', {}),
                    doc.get('vector', [])
                ])
            
            # Insert data in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, collection.insert, data)
            await loop.run_in_executor(None, collection.flush)
            
            self.logger.info(f"Inserted {len(documents)} documents into {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to insert documents: {e}")
            raise
    
    async def search_vectors(self, collection_name: str, query_vector: List[float], 
                      top_k: int = 10, metric_type: str = "COSINE") -> List[Dict]:
        """Search for similar vectors"""
        try:
            collection = Collection(collection_name)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, collection.load)
            
            search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
            
            # Search in thread pool
            results = await loop.run_in_executor(
                None,
                lambda: collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["content", "metadata"]
                )
            )
            
            # Format results
            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    "id": hit.id,
                    "content": hit.entity.get("content"),
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.score
                })
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Failed to search vectors: {e}")
            return []
    
    async def search_intent(self, query_vector: List[float], top_k: int = 1) -> Optional[str]:
        """Search for intent classification"""
        try:
            collection = Collection("intent_queries")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, collection.load)
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # Search in thread pool
            results = await loop.run_in_executor(
                None,
                lambda: collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["intent_label"]
                )
            )
            
            if results[0]:
                return results[0][0].entity.get("intent_label")
            return None
        except Exception as e:
            self.logger.error(f"Failed to search intent: {e}")
            return None
    
    async def close(self):
        """Close connection to Milvus"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, connections.disconnect, self.connection_alias)
            self.logger.info("Milvus connection closed")
        except Exception as e:
            self.logger.error(f"Error closing Milvus connection: {e}")
    
    async def __aenter__(self):
        await self.connect()
        await self.create_knowledge_base_collection()
        await self.create_intent_queries_collection()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
