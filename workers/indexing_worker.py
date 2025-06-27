from tools_services import EmbeddingTool
from database import MilvusManager
import pandas as pd
from typing import List, Dict, Any
import os
import logging

class IndexingWorker:
    def __init__(self):
        self.embedding_tool = EmbeddingTool()
        self.milvus_manager = None
        self.logger = logging.getLogger(__name__)
    
    def create_chunks_from_csv(self, csv_file_path: str) -> List[Dict[str, Any]]:
        """Create text chunks from CSV knowledge base"""
        try:
            df = pd.read_csv(csv_file_path)
            chunks = []
            
            for index, row in df.iterrows():
                # Create text chunk from relevant columns
                chunk_parts = []
                
                # Drug name
                if 'drug_name' in row and pd.notna(row['drug_name']):
                    chunk_parts.append(f"Thuốc: {row['drug_name']}")
                
                # Indication
                if 'indication' in row and pd.notna(row['indication']):
                    chunk_parts.append(f"Chỉ định: {row['indication']}")
                
                # Mechanism
                if 'mechanism' in row and pd.notna(row['mechanism']):
                    chunk_parts.append(f"Cơ chế: {row['mechanism']}")
                
                # Side effects
                if 'side_effects' in row and pd.notna(row['side_effects']):
                    chunk_parts.append(f"Tác dụng phụ: {row['side_effects']}")
                
                # Interactions
                if 'interactions' in row and pd.notna(row['interactions']):
                    chunk_parts.append(f"Tương tác: {row['interactions']}")
                
                # Contraindications
                if 'contraindications' in row and pd.notna(row['contraindications']):
                    chunk_parts.append(f"Chống chỉ định: {row['contraindications']}")
                
                if chunk_parts:
                    content = '. '.join(chunk_parts)
                    
                    # Create metadata
                    metadata = {
                        'source': 'knowledge_base.csv',
                        'row_index': int(index),
                        'drug_name': str(row.get('drug_name', '')),
                        'type': 'drug_information'
                    }
                    
                    chunks.append({
                        'content': content,
                        'metadata': metadata
                    })
            
            self.logger.info(f"Created {len(chunks)} chunks from CSV")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to create chunks from CSV: {e}")
            return []
    
    async def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for chunks"""
        try:
            # Extract content for embedding
            contents = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.embedding_tool.embed_chunks(contents)
            
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
            
            self.logger.info(f"Created embeddings for {len(documents)} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to create embeddings: {e}")
            return []
    
    async def insert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Insert documents into Milvus"""
        try:
            if self.milvus_manager is None:
                self.milvus_manager = MilvusManager()
                await self.milvus_manager.connect()
            
            # Insert documents in batches
            batch_size = 100
            total_inserted = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                await self.milvus_manager.insert_documents("knowledge_base", batch)
                total_inserted += len(batch)
                self.logger.info(f"Inserted batch {i//batch_size + 1}, total: {total_inserted}")
            
            self.logger.info(f"Successfully inserted {total_inserted} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert documents: {e}")
            return False
    
    async def index_knowledge_base(self, csv_file_path: str) -> Dict[str, Any]:
        """Main method to index knowledge base"""
        try:
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
            
            # Step 1: Create chunks from CSV
            chunks = self.create_chunks_from_csv(csv_file_path)
            if not chunks:
                return {'success': False, 'message': 'No chunks created from CSV'}
            
            # Step 2: Create embeddings
            documents = await self.create_embeddings(chunks)
            if not documents:
                return {'success': False, 'message': 'No embeddings created'}
            
            # Step 3: Insert into Milvus
            success = await self.insert_documents(documents)
            
            if success:
                return {
                    'success': True,
                    'message': f'Successfully indexed {len(documents)} documents',
                    'document_count': len(documents)
                }
            else:
                return {'success': False, 'message': 'Failed to insert documents'}
                
        except Exception as e:
            self.logger.error(f"Indexing failed: {e}")
            return {'success': False, 'message': f'Indexing failed: {str(e)}'}
    
    async def close(self):
        """Close connections"""
        if self.milvus_manager:
            await self.milvus_manager.close()
