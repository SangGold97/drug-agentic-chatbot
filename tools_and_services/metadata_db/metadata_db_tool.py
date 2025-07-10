from database.postgres_manager import PostgresManager
from typing import List, Dict, Any
from loguru import logger
import os
from dotenv import load_dotenv
load_dotenv()


class MetadataDBTool:
    def __init__(self):
        self.postgres_manager = PostgresManager()
        self.limit_conversations = int(os.getenv('LIMIT_CONVERSATIONS', 3))
    
    async def connect(self):
        """Connect to Postgres database"""
        try:
            await self.postgres_manager.connect()
            await self.postgres_manager.create_tables("conversations")
            await self.postgres_manager.create_tables("personal_information")
            logger.info("Connected to Postgres database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Postgres database: {e}")
            raise

    async def save_conversation(self, user_id: str, 
                                conversation_id: str, 
                                query: str, answer: str) -> Dict[str, Any]:
        """
        Save conversation to database
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            query: User's query
            answer: System's answer
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:            
            # Get next turn number
            turn = await self.postgres_manager.get_next_turn(user_id, conversation_id)
            
            # Save conversation
            await self.postgres_manager.save_conversation(user_id, conversation_id, turn, query, answer)
            
            logger.info(f"Conversation saved successfully for user {user_id}")
            return {"status": "success", 
                    "message": f"Conversation saved for user {user_id}, conversation id {conversation_id}, turn {turn}"}
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return {"status": "error", "message": str(e)}

    async def get_conversation_history(self, user_id: str, conversation_id: str) -> List[Dict]:
        """
        Get conversation history for a user
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            
        Returns:
            List[Dict]: List of conversation records
        """
        try:
            # Get conversation history
            history = await self.postgres_manager.get_conversation_history(
                user_id, conversation_id, self.limit_conversations
            )
            
            logger.info(f"Retrieved {len(history)} conversation records for user {user_id}")
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def close(self):
        """Close database connection"""
        try:
            await self.postgres_manager.close()
            logger.info("Postgres database connection closed successfully")
        except Exception as e:
            logger.error(f"Failed to close Postgres database connection: {e}")
            raise

    def health_check(self) -> Dict[str, str]:
        """Health check for metadata database tool"""
        try:
            return {"status": "healthy", "message": "Metadata database tool is ready"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}