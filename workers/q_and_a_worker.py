from tools_and_services.llm_services import LLMService
from database import PostgresManager
from typing import List, Dict, AsyncGenerator
import logging

class QAndAWorker:
    def __init__(self):
        self.llm_service = LLMService()
        self.postgres_manager = None
        self.logger = logging.getLogger(__name__)
    
    async def get_chat_history(self, user_id: str, conversation_id: str, limit: int = 5) -> List[Dict]:
        """Get chat history from database"""
        try:
            if self.postgres_manager is None:
                self.postgres_manager = PostgresManager()
                await self.postgres_manager.connect()
            
            history = await self.postgres_manager.get_conversation_history(
                user_id, conversation_id, limit
            )
            
            # Convert to expected format
            formatted_history = []
            for item in history:
                formatted_history.append({
                    'query': item.get('query', ''),
                    'answer': item.get('answer', ''),
                    'created_at': item.get('created_at')
                })
            
            self.logger.info(f"Retrieved {len(formatted_history)} history items")
            return formatted_history
            
        except Exception as e:
            self.logger.error(f"Failed to get chat history: {e}")
            return []
    
    async def generate_answer(self, original_query: str, context: str, user_id: str, conversation_id: str) -> str:
        """Generate final answer using context and chat history"""
        try:
            # Get chat history
            chat_history = await self.get_chat_history(user_id, conversation_id)
            
            # Generate answer
            answer = await self.llm_service.generate_answer(
                original_query, context, chat_history
            )
            
            self.logger.info("Generated answer successfully")
            return answer
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này. Vui lòng thử lại sau."
    
    async def generate_streaming_answer(self, original_query: str, context: str, user_id: str, conversation_id: str) -> AsyncGenerator[str, None]:
        """Generate streaming answer (simulated for now)"""
        try:
            # Get full answer first
            full_answer = await self.generate_answer(original_query, context, user_id, conversation_id)
            
            # Simulate streaming by yielding chunks
            words = full_answer.split()
            chunk_size = 3  # Words per chunk
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += ' '
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Streaming answer generation failed: {e}")
            yield "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời."
    
    async def generate_general_response(self, query: str) -> str:
        """Generate response for non-medical queries"""
        try:
            response = await self.llm_service.generate_general_response(query)
            self.logger.info("Generated general response")
            return response
        except Exception as e:
            self.logger.error(f"General response generation failed: {e}")
            return self.llm_service.prompts.general_prompt(query)
    
    async def close(self):
        """Close database connections"""
        if self.postgres_manager:
            await self.postgres_manager.close()
