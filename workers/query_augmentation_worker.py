from tools_and_services.llm_services import LLMService
from typing import Dict, List
import logging

class QueryAugmentationWorker:
    def __init__(self):
        self.llm_service = LLMService()
        self.logger = logging.getLogger(__name__)
    
    async def augment_query(self, original_query: str) -> Dict[str, any]:
        """Generate structured query and augmented queries"""
        try:
            result = await self.llm_service.query_augmentation(original_query)
            
            # Validate and clean results
            structured_query = result.get('structured_query', original_query)
            augmented_queries = result.get('augmented_queries', [])
            
            # Ensure we have valid augmented queries
            if not augmented_queries:
                augmented_queries = [original_query]
            
            # Limit to max 3 augmented queries
            augmented_queries = augmented_queries[:3]
            
            self.logger.info(f"Generated {len(augmented_queries)} augmented queries")
            
            return {
                'original_query': original_query,
                'structured_query': structured_query,
                'augmented_queries': augmented_queries
            }
            
        except Exception as e:
            self.logger.error(f"Query augmentation failed: {e}")
            # Fallback to original query
            return {
                'original_query': original_query,
                'structured_query': original_query,
                'augmented_queries': [original_query]
            }
