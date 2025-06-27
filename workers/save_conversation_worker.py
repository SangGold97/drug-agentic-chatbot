from database import PostgresManager
from typing import Optional
import logging

class SaveConversationWorker:
    def __init__(self):
        self.postgres_manager = None
        self.logger = logging.getLogger(__name__)
    
    async def save_conversation(self, user_id: str, conversation_id: str, query: str, answer: str) -> bool:
        """Save conversation to database"""
        try:
            if self.postgres_manager is None:
                self.postgres_manager = PostgresManager()
                await self.postgres_manager.connect()
            
            await self.postgres_manager.save_conversation(user_id, conversation_id, query, answer)
            self.logger.info(f"Saved conversation for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation: {e}")
            return False
    
    async def close(self):
        """Close database connection"""
        if self.postgres_manager:
            await self.postgres_manager.close()
