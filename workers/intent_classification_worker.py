from tools_services import EmbeddingTool, VectorSearchTool
from typing import Optional
import logging

class IntentClassificationWorker:
    def __init__(self):
        self.embedding_tool = EmbeddingTool()
        self.vector_search_tool = VectorSearchTool()
        self.logger = logging.getLogger(__name__)
        
        # Medical domain keywords for fallback classification
        self.medical_keywords = [
            'thuốc', 'drug', 'medication', 'medicine', 'bệnh', 'disease', 'illness',
            'gene', 'genetic', 'tương tác', 'interaction', 'tác dụng phụ', 'side effect',
            'liều dùng', 'dosage', 'chống chỉ định', 'contraindication', 'điều trị', 'treatment'
        ]
    
    async def classify_intent(self, query: str) -> str:
        """Classify user query intent"""
        try:
            # First try vector search for intent classification
            intent = await self.vector_search_tool.search_intent(query)
            
            if intent and intent != "medical":
                self.logger.info(f"Intent classified as: {intent}")
                return intent
            
            # Fallback to keyword-based classification
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in self.medical_keywords):
                self.logger.info("Intent classified as: medical (keyword-based)")
                return "medical"
            else:
                self.logger.info("Intent classified as: general")
                return "general"
                
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            # Default to medical for safety
            return "medical"
    
    async def is_medical_query(self, query: str) -> bool:
        """Check if query is related to medical domain"""
        intent = await self.classify_intent(query)
        return intent == "medical"
