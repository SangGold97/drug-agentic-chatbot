from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import os
from dotenv import load_dotenv
from loguru import logger
import asyncio
from time import time

load_dotenv()

class EmbeddingTool:
    def __init__(self):
        self.model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.model = None
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    
    async def load_model(self):
        """Load embedding model"""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")

                # Create cache directory if it doesn't exist
                os.makedirs(self.cache_dir, exist_ok=True)

                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, 
                    lambda: SentenceTransformer(
                        self.model_name, 
                        cache_folder=self.cache_dir
                    )
                )
                logger.info("Embedding model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    async def generate_embedding(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text(s)"""
        if self.model is None:
            await self.load_model()
        
        try:
            # Run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, convert_to_tensor=False)
            )

            # Normalize embeddings to unit length
            embeddings_list = embeddings.tolist()
            normalized_embeddings = self.vector_norm(embeddings_list)
            return normalized_embeddings
        
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return [[] for _ in texts]

    async def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model is None:
            await self.load_model()
        return self.model.get_sentence_embedding_dimension()

    def vector_norm(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize vectors to unit length (L2 normalization)"""
        try:
            normalized_embeddings = []
            for embedding in embeddings:
                # Convert to numpy array for easier computation
                vec = np.array(embedding)
                # Calculate L2 norm
                norm = np.linalg.norm(vec)
                # Avoid division by zero
                if norm > 0:
                    normalized_vec = vec / norm
                else:
                    normalized_vec = vec
                normalized_embeddings.append(normalized_vec.tolist())
            return normalized_embeddings
        
        except Exception as e:
            logger.error(f"Failed to normalize vectors: {e}")
            return embeddings


# Test the EmbeddingTool
if __name__ == "__main__":
    async def test_embedding_tool():
        start = time()
        embedding_tool = EmbeddingTool()
        await embedding_tool.load_model()
        print(f"Embedding model loaded successfully in {(time() - start):.2f}s")
        
        # Get embedding dimension
        dimension = await embedding_tool.get_embedding_dimension()
        print(f"Embedding dimension: {dimension}")
        
        start = time()
        
        # Example texts for medical domain
        texts = [
            "Metformin is a first-line medication for type 2 diabetes management.",
            "Insulin therapy is required for type 1 diabetes patients.",
            "Regular exercise helps control blood glucose levels.",
            "Hypertension is a common comorbidity in diabetic patients.",
            "Aspirin is used for cardiovascular disease prevention.",
            "The liver produces glucose through gluconeogenesis."
        ]
        
        print("Testing embedding generation...")
        embeddings = await embedding_tool.generate_embedding(texts)
        
        print(f"\nGenerated embeddings for {len(texts)} texts")
        print(f"Embedding dimension: {len(embeddings[0]) if embeddings and embeddings[0] else 0}")
        
        # Show sample embeddings (first 5 dimensions)
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            if embedding:
                print(f"\nText {i+1}: {text[:50]}...")
                print(f"Embedding (first 5 dims): {embedding[:5]}")
                print(f"Vector norm: {np.linalg.norm(embedding):.4f}")
        
        inference_time = time() - start
        print(f"\nTotal inference time: {inference_time:.2f}s")
        print(f"Average time per text: {inference_time/len(texts):.4f}s")

    asyncio.run(test_embedding_tool())
