from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from loguru import logger
import asyncio

load_dotenv()

class MilvusManager:
    def __init__(self):
        self.host = os.getenv('MILVUS_HOST', 'localhost')
        self.port = int(os.getenv('MILVUS_PORT', 19530))
        self.db_name = os.getenv('MILVUS_DB', 'drug_chatbot')
        self.vector_dim = int(os.getenv('VECTOR_DIMENSION', 1024))
        self.connection_alias = "default"
    
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
            logger.info("Connected to Milvus database")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    async def create_knowledge_base_collection(self) -> Collection | None:
        """Create knowledge base collection"""
        try:
            collection_name = "knowledge_base"
            
            loop = asyncio.get_event_loop()
            
            # Check if collection exists
            exists = await loop.run_in_executor(None, utility.has_collection, collection_name)
            if exists:
                logger.info(f"Collection {collection_name} already exists")
                return Collection(collection_name)
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
            ]
            
            schema = CollectionSchema(fields, "Knowledge base collection")
            collection = await loop.run_in_executor(
                None, lambda: Collection(collection_name, schema)
            )

            # Create index for vector field
            index_params = {
                "metric_type": "IP",
                "index_type": "FLAT"
            }
            await loop.run_in_executor(None, collection.create_index, "vector", index_params)
            
            logger.info(f"Created collection {collection_name}")
            return collection
        
        except Exception as e:
            logger.error(f"Failed to create knowledge base collection: {e}")
            return None

    async def create_intent_queries_collection(self) -> Collection | None:
        """Create intent queries collection"""
        try:
            collection_name = "intent_queries"
            loop = asyncio.get_event_loop()
            
            # Check if collection exists
            exists = await loop.run_in_executor(None, utility.has_collection, collection_name)
            if exists:
                logger.info(f"Collection {collection_name} already exists")
                return Collection(collection_name)
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="intent_label", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
            ]
            
            schema = CollectionSchema(fields, "Intent queries collection")
            collection = await loop.run_in_executor(
                None,
                lambda: Collection(collection_name, schema)
            )
            
            # Create index for vector field
            index_params = {
                "metric_type": "IP",
                "index_type": "FLAT"
            }
            await loop.run_in_executor(None, collection.create_index, "vector", index_params)
            
            logger.info(f"Created collection {collection_name}")
            return collection
        
        except Exception as e:
            logger.error(f"Failed to create intent queries collection: {e}")
            return None
    
    async def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        """Insert documents into collection"""
        try:
            # Prepare data for insertion
            if collection_name == "knowledge_base":
                collection = await self.create_knowledge_base_collection()
                data = [
                    [doc.get('content', '') for doc in documents],      # content field
                    [doc.get('metadata', {}) for doc in documents],     # metadata field  
                    [doc.get('vector', []) for doc in documents]        # vector field
                ]
            elif collection_name == "intent_queries":
                collection = await self.create_intent_queries_collection()
                data = [
                    [doc.get('query', '') for doc in documents],        # query field
                    [doc.get('intent_label', '') for doc in documents], # intent_label field
                    [doc.get('vector', []) for doc in documents]        # vector field
                ]
            else:
                raise ValueError(f"Unknown collection: {collection_name}")
            
            # Insert data in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, collection.insert, data)
            await loop.run_in_executor(None, collection.flush)
            
            logger.info(f"Inserted {len(documents)} documents into {collection_name}")
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
    
    async def search_vector(self, query_vector: List[float], collection_name: str,
                            top_k: int = 10, metric_type: str = "IP") -> List[Dict]:
        """Search for similar vectors"""
        try:
            collection = Collection(collection_name)
            search_params = {"metric_type": metric_type}
            loop = asyncio.get_event_loop()

            # Search in knowledge base collection
            if collection_name == "knowledge_base":            
                await loop.run_in_executor(None, collection.load)
                
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
                        "content": hit.entity.get("content"),
                        "metadata": hit.entity.get("metadata"),
                        "score": hit.score
                    })
                return formatted_results
            
            elif collection_name == "intent_queries":
                await loop.run_in_executor(None, collection.load)
                
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
                
                # Format results
                formatted_results = []
                for hit in results[0]:
                    formatted_results.append({
                        "intent_label": hit.entity.get("intent_label"),
                        "score": hit.score
                    })
                return formatted_results
            
            else:
                logger.warning(f"Unknown collection: {collection_name}")
                return []

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics for all collections"""
        try:
            # List all collections
            collection_names = utility.list_collections()
            stats = {}
            
            for collection_name in collection_names:
                try:
                    collection = Collection(collection_name)
                    # Load collection to get accurate count
                    collection.load()
                    # Get number of entities (documents)
                    stats[collection_name] = collection.num_entities
                except Exception as e:
                    logger.warning(f"Failed to get stats for collection {collection_name}: {e}")
                    stats[collection_name] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    def delete_collection(self, collection_name: str) -> Dict[str, str]:
        """Delete a collection if it exists"""
        try:
            # Check if collection exists
            if utility.has_collection(collection_name):
                # Drop the collection
                utility.drop_collection(collection_name)
                logger.info(f"Successfully deleted collection: {collection_name}")
                return {"status": "success", "message": f"Collection '{collection_name}' deleted successfully"}
            else:
                logger.warning(f"Collection '{collection_name}' does not exist")
                return {"status": "success", "message": f"Collection '{collection_name}' does not exist"}

        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return {"status": "error", "message": f"Failed to delete collection: {str(e)}"}

    async def close(self):
        """Close connection to Milvus"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, connections.disconnect, self.connection_alias)
            logger.info("Milvus connection closed")
        except Exception as e:
            logger.error(f"Error closing Milvus connection: {e}")

