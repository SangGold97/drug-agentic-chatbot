import asyncpg
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import logging

load_dotenv()

class PostgresManager:
    def __init__(self):
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = int(os.getenv('POSTGRES_PORT', 5432))
        self.database = os.getenv('POSTGRES_DB', 'drug_chatbot')
        self.user = os.getenv('POSTGRES_USER', 'postgres')
        self.password = os.getenv('POSTGRES_PASSWORD', 'password')
        self.connection = None
        self.logger = logging.getLogger(__name__)
    
    async def connect(self):
        """Establish database connection"""
        try:
            self.connection = await asyncpg.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.logger.info("Connected to PostgreSQL database")
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    async def create_tables(self):
        """Create necessary tables if they don't exist"""
        create_conversations_table = """
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            conversation_id VARCHAR(255) NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_user_conversation 
        ON conversations (user_id, conversation_id);
        """
        
        try:
            await self.connection.execute(create_conversations_table)
            self.logger.info("Tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise
    
    async def save_conversation(self, user_id: str, conversation_id: str, query: str, answer: str):
        """Save conversation to database"""
        insert_query = """
        INSERT INTO conversations (user_id, conversation_id, query, answer)
        VALUES ($1, $2, $3, $4)
        """
        
        try:
            await self.connection.execute(insert_query, user_id, conversation_id, query, answer)
            self.logger.info(f"Conversation saved for user {user_id}")
        except Exception as e:
            self.logger.error(f"Failed to save conversation: {e}")
            raise
    
    async def get_conversation_history(self, user_id: str, conversation_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a user"""
        select_query = """
        SELECT query, answer, created_at
        FROM conversations
        WHERE user_id = $1 AND conversation_id = $2
        ORDER BY created_at DESC
        LIMIT $3
        """
        
        try:
            rows = await self.connection.fetch(select_query, user_id, conversation_id, limit)
            return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            self.logger.info("PostgreSQL connection closed")
    
    async def __aenter__(self):
        await self.connect()
        await self.create_tables()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
