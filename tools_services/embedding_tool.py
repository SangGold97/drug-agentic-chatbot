from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import os
from dotenv import load_dotenv
import logging
import asyncio

load_dotenv()

class EmbeddingTool:
    def __init__(self):
        self.model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    async def load_model(self):
        """Load embedding model"""
        if self.model is None:
            try:
                self.logger.info(f"Loading embedding model: {self.model_name}")
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(None, SentenceTransformer, self.model_name)
                self.logger.info("Embedding model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                raise
    
    async def embed_text(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text(s)"""
        if self.model is None:
            await self.load_model()
        
        try:
            # Run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            if isinstance(texts, str):
                embeddings = await loop.run_in_executor(
                    None, 
                    lambda: self.model.encode(texts, convert_to_tensor=False)
                )
                return embeddings.tolist()
            else:
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: self.model.encode(texts, convert_to_tensor=False)
                )
                return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            return [] if isinstance(texts, str) else [[] for _ in texts]
    
    async def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text chunks"""
        return await self.embed_text(chunks)
    
    async def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model is None:
            await self.load_model()
        return self.model.get_sentence_embedding_dimension()
