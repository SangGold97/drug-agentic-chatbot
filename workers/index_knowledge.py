import pandas as pd
from typing import List, Dict, Any
import os
import httpx
import asyncio
from loguru import logger

class IndexingWorker:
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize Retriever with base URL for the tools API
        
        Args:
            base_url: Base URL for the tools and services API
        """
        self.base_url = base_url

    def _create_chunks_from_csv(self, csv_file_path: str) -> List[Dict[str, Any]]:
        """Create text chunks from CSV knowledge base"""
        try:
            df = pd.read_csv(csv_file_path)
            chunks = []
            
            for id, row in df.iterrows():
                # Create text chunk from relevant columns
                chunk_parts = []
                
                # Drug name
                if 'name' in row and pd.notna(row['name']):
                    chunk_parts.append(f"Hoạt chất thuốc {row['name']}")
                
                # Group
                if 'group' in row and pd.notna(row['group']):
                    chunk_parts.append(f"Thuộc nhóm: {row['group']}")

                # Related diseases
                if 'related_diseases' in row and pd.notna(row['related_diseases']):
                    chunk_parts.append(f"Thuốc này chỉ định cho các bệnh: {row['related_diseases']}")

                # Related gene
                if 'related_gene' in row and row['related_gene'] not in [None, '']:
                    chunk_parts.append(f"Trong báo cáo PGx của Genestory, liên quan đến gene: {row['related_gene']}")
                
                # Product names
                if 'product_names' in row and pd.notna(row['product_names']):
                    chunk_parts.append(f"Một số sản phẩm chứa hoạt chất thuốc: {row['product_names']}")
                
                if chunk_parts:
                    content = '\n'.join(chunk_parts)
                    
                    # Create metadata
                    metadata = {
                        'category': row['category'] if 'category' in row and pd.notna(row['category']) else '',
                        'recommendation': row['recommendation'] if 'recommendation' in row and pd.notna(row['recommendation']) else '',
                        'description': row['description'] if 'description' in row and pd.notna(row['description']) else '',
                    }
                    
                    chunks.append({
                        'content': content,
                        'metadata': metadata
                    })
            
            logger.info(f"Created {len(chunks)} chunks from CSV")
            # logger.info(f"Sample content:\n{chunks[0]['content']}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to create chunks from CSV: {e}")
            return []
    
    async def _create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for chunks"""
        try:
            # Extract content for embedding
            contents = [chunk['content'] for chunk in chunks]
            
            # Call embedding API with increased timeout
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/embedding/generate_embedding",
                    json={"texts": contents}
                )
                response.raise_for_status()
                embeddings = response.json()["embeddings"]
            
            # Add embeddings to chunks
            documents = []
            for i, chunk in enumerate(chunks):
                if i < len(embeddings) and embeddings[i]:
                    document = {
                        'content': chunk['content'],
                        'metadata': chunk['metadata'],
                        'vector': embeddings[i]
                    }
                    documents.append(document)

            logger.info(f"Created embeddings for {len(documents)} documents")
            # logger.info(f"Sample content and embedding:\n{documents[0]['content']}\n{documents[0]['vector'][:10]}...")  # Show first 10 elements of vector
            return documents
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return []
    
    async def _insert_chunks(self, documents: List[Dict[str, Any]]) -> bool:
        """Insert documents into Milvus via vector_db API"""
        try:
            # Call vector_db insert API with increased timeout
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/vector_db/insert",
                    json={
                        "collection_name": "knowledge_base",
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
        """Main method to index knowledge base"""
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
                    'message': f'Successfully indexed {len(documents)} documents',
                    'document_count': len(documents)
                }
            else:
                return {'success': False, 'message': 'Failed to insert documents'}
                
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return {'success': False, 'message': f'Indexing failed: {str(e)}'}
        
    
    async def delete_collection(self, collection_name: str) -> Dict[str, Any]:
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
        
    async def get_stats_collection(self):
        """Get statistics of a collection from Milvus via vector_db API"""
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
    """Test function for the IndexingWorker class"""
    # Initialize indexing worker
    worker = IndexingWorker()

    async def get_stats():
        # Get stats of collections
        stats_result = await worker.get_stats_collection()
        logger.info("Collection stats:")
        if stats_result['success']:
            for collection, num in stats_result['stats'].items():
                logger.info(f"  - Collection: {collection} | Number of documents: {num}")

    await get_stats()

    # CSV file path - adjust this to your actual CSV file
    csv_file_path = "../knowledge_base/knowledge_base.csv"
    
    logger.info(f"Testing IndexingWorker with CSV file: {csv_file_path}")
    
    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found: {csv_file_path}")
        logger.info("Available CSV files in knowledge_base directory:")
        kb_dir = "../knowledge_base"
        if os.path.exists(kb_dir):
            csv_files = [f for f in os.listdir(kb_dir) if f.endswith('.csv')]
            for csv_file in csv_files:
                logger.info(f"  - {csv_file}")
        return
    
    try:
        # Run indexing process
        logger.info("Starting indexing process...")
        result = await worker.run(csv_file_path)
        
        # Display results
        logger.info("Indexing completed!")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Message: {result['message']}")
        
        if 'document_count' in result:
            logger.info(f"Documents indexed: {result['document_count']}")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")

    await get_stats()

    # Delete collection for cleanup
    # delete_result = await worker.delete_collection("knowledge_base")
    # logger.info(f"Delete collection result: {delete_result['message']}")

    # await get_stats()

if __name__ == "__main__":
    asyncio.run(main())
