import pandas as pd
from typing import List, Dict, Any
import os
import httpx
import asyncio
from loguru import logger

class IndexIntent:
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize IndexIntent with base URL for the tools API
        
        Args:
            base_url: Base URL for the tools and services API
        """
        self.base_url = base_url

    def _create_chunks_from_csv(self, csv_file_path: str) -> List[Dict[str, str]]:
        """Create chunks from intent_queries.csv"""
        try:
            df = pd.read_csv(csv_file_path)
            # logger.info(f"Number of label medical queries: {df[df['label'] == 'medical'].shape[0]}")
            # logger.info(f"Number of label non-medical queries: {df[df['label'] != 'medical'].shape[0]}")
            chunks = []
            
            for _, row in df.iterrows():
                if pd.notna(row['query']) and pd.notna(row['label']):
                    chunks.append({
                        'query': row['query'],
                        'intent_label': row['label']
                    })
            
            logger.info(f"Created {len(chunks)} chunks from CSV")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to create chunks from CSV: {str(e)}")
            return []

    async def _create_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Create embeddings for chunks"""
        try:
            # Extract queries for embedding
            queries = [chunk['query'] for chunk in chunks]
            
            # Call embedding API
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/embedding/generate_embedding",
                    json={"texts": queries}
                )
                response.raise_for_status()
                embeddings = response.json()["embeddings"]
            
            # Create documents with embeddings
            documents = []
            for i, chunk in enumerate(chunks):
                if i < len(embeddings) and embeddings[i]:
                    document = {
                        'query': chunk['query'],
                        'intent_label': chunk['intent_label'],
                        'vector': embeddings[i]
                    }
                    documents.append(document)

            logger.info(f"Created embeddings for {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return []

    async def _insert_chunks(self, documents: List[Dict[str, Any]]) -> bool:
        """Insert documents into Milvus via vector_db API"""
        try:
            # Call vector_db insert API
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/vector_db/insert",
                    json={
                        "collection_name": "intent_queries",
                        "documents": documents
                    }
                )
                response.raise_for_status()
                result = response.json()
            
            if result["status"] == "success":
                logger.info(f"Successfully inserted {len(documents)} documents")
                return True
            else:
                logger.error(f"Failed to insert documents: {result.get('message', 'Unknown error')}")
                return False         
            
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return False

    async def run(self, csv_file_path: str) -> Dict[str, Any]:
        """Main method to index intent queries"""
        try:
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
            
            # Step 1: Create chunks from CSV
            chunks = self._create_chunks_from_csv(csv_file_path)
            if not chunks:
                return {'success': False, 'message': 'No chunks created from CSV'}
            
            # Step 2: Create embeddings
            documents = await self._create_embeddings(chunks)
            if not documents:
                return {'success': False, 'message': 'No embeddings created'}
            
            # Step 3: Insert into Milvus
            success = await self._insert_chunks(documents)
            
            if success:
                return {
                    'success': True,
                    'message': f'Successfully indexed {len(documents)} intent queries',
                    'document_count': len(documents)
                }
            else:
                return {'success': False, 'message': 'Failed to insert documents'}
                
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return {'success': False, 'message': f'Indexing failed: {str(e)}'}

    async def delete_collection(self, collection_name: str = 'intent_queries') -> Dict[str, Any]:
        """Delete a collection from Milvus via vector_db API"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.base_url}/vector_db/delete_collection",
                    json={"collection_name": collection_name}
                )
                response.raise_for_status()
                result = response.json()
                return {
                    'success': result.get('status') == 'success',
                    'message': result.get('message', f'Collection {collection_name} deleted successfully')
                }
                
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return {'success': False, 'message': f'Failed to delete collection: {str(e)}'}

    async def get_stats_collection(self) -> Dict[str, Any]:
        """Get statistics of collections from Milvus via vector_db API"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/vector_db/stats"
                )
                response.raise_for_status()
                result = response.json()
                return {
                    'success': result.get('status') == 'success',
                    'stats': result.get('stats', {})
                }

        except Exception as e:
            logger.error(f"Failed to get stats of collections: {e}")
            return {'success': False, 'message': f'Failed to get stats: {str(e)}'}


async def main():
    """Test function for the IndexIntent class"""
    # Initialize indexing worker
    worker = IndexIntent()

    async def get_stats():
        # Get stats of collections
        stats_result = await worker.get_stats_collection()
        logger.info("Collection stats:")
        if stats_result['success']:
            for collection, num in stats_result['stats'].items():
                logger.info(f"  - Collection: {collection} | Number of documents: {num}")

    await get_stats()

    await worker.delete_collection()

    # CSV file path
    csv_file_path = "../intent_queries/intent_queries.csv"
    
    logger.info(f"Testing IndexIntent with CSV file: {csv_file_path}")
    
    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found: {csv_file_path}")
        return
    
    try:
        # Run indexing process
        logger.info("Starting intent indexing process...")
        result = await worker.run(csv_file_path)
        
        # Display results
        logger.info("Intent indexing completed!")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Message: {result['message']}")
        
        if 'document_count' in result:
            logger.info(f"Intent queries indexed: {result['document_count']}")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")

    await get_stats()


if __name__ == "__main__":
    asyncio.run(main())