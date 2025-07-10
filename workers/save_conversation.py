import httpx
import json
import asyncio
from typing import Dict, List, Any
from loguru import logger

class SaveConversation:
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize SaveConversation with base URL for the tools API

        Args:
            base_url: Base URL for the tools and services API
        """
        self.base_url = base_url
    
    async def run(self, user_id: str, conversation_id: str, query: str, answer: str) -> Dict[str, str]:
        """
        Save conversation to database

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            query: User's query
            answer: System's answer

        Returns:
            dict: Result of the save operation
        """
        try:
            url = f"{self.base_url}/metadata_db/save_conversation"
            payload = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "query": query,
                "answer": answer
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Conversation saved successfully: {result}")
                return {
                    "success": result.get('status') == 'success',
                    "message": result.get('message', 'Conversation saved successfully')
                }
            
        except Exception as e:
            logger.error(f"An error occurred while saving conversation: {str(e)}")
            return {
                "success": False,
                "message": "Failed to save conversation"
            }

async def main():
    """
    Test function for SaveConversation
    """
    # Initialize SaveConversation instance
    save_conversation = SaveConversation()
    
    # Test data
    test_user_id = "user_123"
    test_conversation_id = "conv_456"
    test_query = "What are the side effects of Aspirin?"
    test_answer = "Aspirin side effects include stomach irritation, bleeding risk, and allergic reactions."
    
    try:
        # Test the run method
        result = await save_conversation.run(
            user_id=test_user_id,
            conversation_id=test_conversation_id,
            query=test_query,
            answer=test_answer
        )
        
        print("Test Result:")
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

