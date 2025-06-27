from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
import logging
import asyncio

load_dotenv()

class RerankTool:
    def __init__(self):
        self.model_name = os.getenv('RERANK_MODEL', 'jinaai/jina-reranker-v2-base-multilingual')
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    async def load_model(self):
        """Load rerank model"""
        if self.model is None:
            try:
                self.logger.info(f"Loading rerank model: {self.model_name}")
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(None, CrossEncoder, self.model_name)
                self.logger.info("Rerank model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load rerank model: {e}")
                raise
    
    async def rerank_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Rerank chunks based on query relevance"""
        if not chunks:
            return chunks
        
        if self.model is None:
            await self.load_model()
        
        try:
            # Prepare query-chunk pairs for reranking
            query_chunk_pairs = []
            for chunk in chunks:
                content = chunk.get('content', '')
                query_chunk_pairs.append([query, content])
            
            # Get relevance scores in thread pool
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                self.model.predict,
                query_chunk_pairs
            )
            
            # Combine chunks with scores and sort
            chunks_with_scores = []
            for i, chunk in enumerate(chunks):
                chunk_copy = chunk.copy()
                chunk_copy['rerank_score'] = float(scores[i])
                chunks_with_scores.append(chunk_copy)
            
            # Sort by rerank score (descending)
            reranked_chunks = sorted(
                chunks_with_scores, 
                key=lambda x: x['rerank_score'], 
                reverse=True
            )
            
            self.logger.info(f"Reranked {len(chunks)} chunks")
            return reranked_chunks
        
        except Exception as e:
            self.logger.error(f"Failed to rerank chunks: {e}")
            # Return original chunks if reranking fails
            return chunks
    
    async def get_top_k_reranked(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Get top-k reranked chunks"""
        reranked_chunks = await self.rerank_chunks(query, chunks)
        return reranked_chunks[:top_k]
