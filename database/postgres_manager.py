import asyncpg
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from loguru import logger
from asyncpg.connection import Connection

load_dotenv()

class PostgresManager:
    def __init__(self):
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = int(os.getenv('POSTGRES_PORT', 5432))
        self.database = os.getenv('POSTGRES_DB', 'drug_chatbot')
        self.user = os.getenv('POSTGRES_USER', 'postgres')
        self.password = os.getenv('POSTGRES_PASSWORD', 'password')
        self.connection: Optional[Connection] = None

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
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def create_tables(self, table_name: str):
        """Create necessary tables if they don't exist"""
        if table_name == "conversations":
            command = """
            CREATE TABLE IF NOT EXISTS conversations (
                user_id VARCHAR(255) NOT NULL,
                conversation_id VARCHAR(255) NOT NULL,
                turn INT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, conversation_id, turn)
            );
            
            CREATE INDEX IF NOT EXISTS idx_user_conversation 
            ON conversations (user_id, conversation_id);
            """
        elif table_name == "personal_information":
            command = """
            CREATE TABLE IF NOT EXISTS personal_information (
                user_id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255),
                age INT,
                gender VARCHAR(50)
            );
            """
        else:
            logger.warning(f"Unknown table name: {table_name}")
            raise ValueError(f"Unknown table name: {table_name}")

        try:
            await self.connection.execute(command)
            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def save_conversation(self, user_id: str, conversation_id: str, turn: int, query: str, answer: str):
        """Save conversation to database"""
        insert_query = """
        INSERT INTO conversations (user_id, conversation_id, turn, query, answer)
        VALUES ($1, $2, $3, $4, $5)
        """
        
        try:
            await self.connection.execute(insert_query, user_id, conversation_id, turn, query, answer)
            logger.info(f"Conversation saved for user {user_id}, conversation {conversation_id}, turn {turn}")
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise
    
    async def get_next_turn(self, user_id: str, conversation_id: str) -> int:
        """Get the next turn number for a conversation"""
        select_query = """
        SELECT COALESCE(MAX(turn), 0) + 1 as next_turn
        FROM conversations
        WHERE user_id = $1 AND conversation_id = $2
        """
        
        try:
            row = await self.connection.fetchrow(select_query, user_id, conversation_id)
            return int(row['next_turn'])
        except Exception as e:
            logger.error(f"Failed to get next turn: {e}")
            return 1
    
    async def get_conversation_history(self, user_id: str, conversation_id: str, limit: int) -> List[Dict]:
        """Get conversation history for a user"""
        select_query = """
        SELECT * FROM (
            SELECT turn, query, answer
            FROM conversations
            WHERE user_id = $1 AND conversation_id = $2
            ORDER BY turn DESC
            LIMIT $3
        ) sub
        ORDER BY turn ASC;
        """
        
        try:
            rows = await self.connection.fetch(select_query, user_id, conversation_id, limit)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            logger.info("PostgreSQL connection closed")

