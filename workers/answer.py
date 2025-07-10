from typing import List, Dict, AsyncGenerator
from loguru import logger
import httpx

class Answer:
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize Answer with base URL for the tools API
        
        Args:
            base_url: Base URL for the tools and services API
        """
        self.base_url = base_url
        self.user_id = None
        self.conversation_id = None
    
    def set_user_info(self, user_id: str, conversation_id: str):
        """Set user and conversation information"""
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    async def _get_conversation_history(self) -> List[Dict]:
        """
        Get conversation history for the current user and conversation
        
        Returns:
            List[Dict]: History with 'query' and 'answer' keys
        """
        if not self.user_id or not self.conversation_id:
            logger.warning("User ID or conversation ID not set")
            return []
        
        try:
            url = f"{self.base_url}/metadata_db/get_conversation_history"
            payload = {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    return result.get("history", [])
                else:
                    logger.error(f"Failed to get conversation history: {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def _parse_context(self, retriever_result: Dict) -> str:
        """
        Parse context from retriever result into formatted text
        
        Args:
            retriever_result: Result from retriever.run() containing 'vector_search' and 'web_search'
            
        Returns:
            str: Formatted context text
        """
        context_parts = []
        
        # Parse vector search results
        context_parts.append("=== THÔNG TIN TỪ BÁO CÁO PGx CỦA GENESTORY DÀNH CHO NGƯỜI DÙNG ===")
        vector_search = retriever_result.get("vector_search", [])
        if vector_search:
            for i, item in enumerate(vector_search, 1):
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                
                context_parts.append(f"\n{i}. {content}")
                
                # Add metadata information if available
                if metadata:
                    if "category" in metadata:
                        context_parts.append(f"- Kết luận: {metadata['category']}")
                    if "recommendation" in metadata:
                        context_parts.append(f"- Khuyến nghị: {metadata['recommendation']}")
                    if "description" in metadata:
                        context_parts.append(f"- Cơ sở khoa học: {metadata['description']}")
        else:
            context_parts.append("Không có thông tin này trong gói PGx của người dùng hoặc người dùng chưa mua gói PGx")

        # Parse web search results
        context_parts.append("\n=== THÔNG TIN TỪ WEB ===")
        web_search = retriever_result.get("web_search", {})
        if web_search:
            for query, results in web_search.items():
                if results:
                    for i, item in enumerate(results, 1):
                        url = item.get("url", "")
                        content = item.get("content", "")
                        
                        # Truncate content if too long
                        if len(content) > 10000:
                            content = content[:10000] + "..."
                        
                        context_parts.append(f"Nguồn URL: {url}")
                        context_parts.append(f"Nội dung: {content}\n")
        else:
            context_parts.append("Không có thông tin")

        return "\n".join(context_parts)
    
    async def run(self, query: str, service_name: str, context: Dict = None) -> Dict:
        """
        Run the answer generation process
        
        Args:
            query: User query
            service_name: Either 'general' or 'answer'
            context: Context from retriever (only needed for 'answer' service)
            
        Returns:
            Dict: Response with 'success', 'query', 'answer', 'message'
        """
        try:
            if service_name == "general":
                # For general service, only need conversation history
                chat_history = await self._get_conversation_history()
                
                # Send request to LLM service
                url = f"{self.base_url}/llm/generate_response"
                payload = {
                    "service_name": "general",
                    "query": query,
                    "chat_history": chat_history
                }
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("response", "")
                        
                        return {
                            "success": True,
                            "query": query,
                            "answer": answer,
                            "message": "Response generated successfully"
                        }
                    else:
                        return {
                            "success": False,
                            "query": query,
                            "answer": "",
                            "message": f"LLM service failed with status {response.status_code}"
                        }
            
            elif service_name == "answer":
                if not context:
                    return {
                        "success": False,
                        "query": query,
                        "answer": "",
                        "message": "Context is required for answer service"
                    }
                
                # Parse context from retriever result
                parsed_context = self._parse_context(context)
                
                # Get conversation history
                chat_history = await self._get_conversation_history()
                
                # Send request to LLM service
                url = f"{self.base_url}/llm/generate_response"
                payload = {
                    "service_name": "answer",
                    "query": query,
                    "context": parsed_context,
                    "chat_history": chat_history
                }
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("response", "")
                        
                        return {
                            "success": True,
                            "query": query,
                            "answer": answer,
                            "message": "Response generated successfully with context"
                        }
                    else:
                        return {
                            "success": False,
                            "query": query,
                            "answer": "",
                            "message": f"LLM service failed with status {response.status_code}"
                        }
            
            else:
                return {
                    "success": False,
                    "query": query,
                    "answer": "",
                    "message": f"Unknown service name: {service_name}"
                }
                
        except Exception as e:
            logger.error(f"Error in Answer.run(): {e}")
            return {
                "success": False,
                "query": query,
                "answer": "",
                "message": f"Error: {str(e)}"
            }


async def main():
    """Test function for the Answer class"""
    import asyncio
    
    # Initialize Answer
    answer = Answer()
    answer.set_user_info("user_123", "conv_456")
    
    logger.info("Testing Answer class...")
    
    # Test 1: General service
    logger.info("Test 1: General service")
    test_query = "Thành phần hóa học của muối hồng là gì? Muối hồng có tác dụng gì với sức khỏe con người?"

    result_general = await answer.run(
        query=test_query,
        service_name="general"
    )
    
    logger.info(f"General service result:")
    logger.info(f"  Success: {result_general['success']}")
    logger.info(f"  Query: {result_general['query']}")
    logger.info(f"  Answer: {result_general['answer']}")
    logger.info(f"  Message: {result_general['message']}")
    
    # Test 2: Answer service with mock context
    logger.info("\nTest 2: Answer service with mock context")
    test_query = "Tôi 28 tuổi, nam, bị bệnh đau dạ dày. Tôi muốn hỏi thuốc aspirin có tác dụng gì? với kiểu gen cyp2d6 của tôi, tôi có nên dùng thuốc này không? gen cyp2d6 tương tác với thuốc này như thế nào?"

    # Mock context similar to retriever output
    mock_context = {
        "vector_search": [
            {
                "content": "Aspirin là một loại thuốc thuộc nhóm NSAID (thuốc chống viêm không steroid) được sử dụng để giảm đau, hạ sốt và chống viêm.",
                "metadata": {
                    "category": "cảnh giác khi sử dụng",
                    "recommendation": "Kiểu gen CYP2D6 của bạn cho thấy bạn chuyển hóa trung bình Aspirin.",
                    "description": "Các con đường chính để chuyển hóa codeine xảy ra ở gan, mặc dù một số chuyển hóa xảy ra ở ruột và não. Từ 0 đến 15% codein bị O-demethyl hóa thành morphin, chất chuyển hóa hoạt tính mạnh nhất. Phản ứng chuyển hóa này được thực hiện bởi CYP2D6."
                }
            }
        ],
        "web_search": {
            "aspirin tác dụng": [
                {
                    "url": "https://example.com/aspirin-info",
                    "content": "Aspirin (acetylsalicylic acid) là một trong những loại thuốc được sử dụng rộng rãi nhất trên thế giới. Nó có tác dụng giảm đau, hạ sốt, chống viêm và chống đông máu..."
                },
                {
                    "url": "https://example.com/aspirin-uses",
                    "content": "Aspirin được sử dụng để điều trị nhiều tình trạng khác nhau, bao gồm đau đầu, đau cơ, viêm khớp và các bệnh tim mạch..."
                }
            ]
        }
    }
    
    result_answer = await answer.run(
        query=test_query,
        service_name="answer",
        context=mock_context
    )
    
    logger.info(f"Answer service result:")
    logger.info(f"  Success: {result_answer['success']}")
    logger.info(f"  Query: {result_answer['query']}")
    logger.info(f"  Answer: {result_answer['answer']}")
    logger.info(f"  Message: {result_answer['message']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


