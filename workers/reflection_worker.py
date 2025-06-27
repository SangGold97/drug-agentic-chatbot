from tools_services.llm_services import LLMService
from tools_services import WebSearchTool
from typing import List, Dict
import os
from dotenv import load_dotenv
import logging

load_dotenv()

class ReflectionWorker:
    def __init__(self):
        self.llm_service = LLMService()
        self.web_search_tool = WebSearchTool()
        self.max_retries = int(os.getenv('MAX_RETRY_REFLECTION', 2))
        self.logger = logging.getLogger(__name__)
    
    async def reflect_on_context(self, structured_query: str, aug_queries: List[str], context: str) -> Dict:
        """Assess if context is sufficient to answer queries"""
        try:
            reflection_result = await self.llm_service.reflection_check(
                structured_query, aug_queries, context
            )
            
            sufficient = reflection_result.get('sufficient', False)
            reasoning = reflection_result.get('reasoning', '')
            follow_up_queries = reflection_result.get('follow_up_queries', [])
            
            self.logger.info(f"Reflection result: sufficient={sufficient}")
            
            return {
                'sufficient': sufficient,
                'reasoning': reasoning,
                'follow_up_queries': follow_up_queries
            }
            
        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            # Conservative fallback - assume insufficient
            return {
                'sufficient': False,
                'reasoning': 'Không thể đánh giá đầy đủ thông tin',
                'follow_up_queries': []
            }
    
    async def enhance_context_with_followup(self, context: str, follow_up_queries: List[str]) -> str:
        """Enhance context with follow-up web search"""
        if not follow_up_queries:
            return context
        
        try:
            # Search additional content
            query_contents = await self.web_search_tool.search_and_fetch(follow_up_queries)
            
            if not query_contents:
                return context
            
            # Summarize additional content
            additional_summaries = []
            for query, contents in query_contents.items():
                if contents:
                    merged_content = '\n\n'.join([item['content'] for item in contents])
                    summary = await self.llm_service.summarize_web_content(query, merged_content)
                    if summary:
                        additional_summaries.append(f"Thông tin bổ sung về '{query}':\n{summary}")
            
            if additional_summaries:
                additional_context = '\n\n'.join(additional_summaries)
                enhanced_context = f"{context}\n\n=== THÔNG TIN BỔ SUNG ===\n{additional_context}"
                self.logger.info(f"Enhanced context with {len(follow_up_queries)} follow-up queries")
                return enhanced_context
            
            return context
            
        except Exception as e:
            self.logger.error(f"Follow-up enhancement failed: {e}")
            return context
    
    async def iterative_reflection(self, structured_query: str, aug_queries: List[str], initial_context: str) -> Dict:
        """Perform iterative reflection with follow-up searches"""
        context = initial_context
        retry_count = 0
        
        while retry_count < self.max_retries:
            # Reflect on current context
            reflection = await self.reflect_on_context(structured_query, aug_queries, context)
            
            if reflection['sufficient']:
                self.logger.info(f"Context sufficient after {retry_count} iterations")
                return {
                    'final_context': context,
                    'sufficient': True,
                    'reasoning': reflection['reasoning'],
                    'retry_count': retry_count
                }
            
            # Try to enhance context with follow-up queries
            follow_up_queries = reflection['follow_up_queries']
            if follow_up_queries:
                context = await self.enhance_context_with_followup(context, follow_up_queries)
                retry_count += 1
                self.logger.info(f"Enhanced context in iteration {retry_count}")
            else:
                # No follow-up queries, stop trying
                break
        
        self.logger.info(f"Reflection completed after {retry_count} iterations, context may be insufficient")
        return {
            'final_context': context,
            'sufficient': False,
            'reasoning': f"Đã thử {retry_count} lần nhưng thông tin có thể chưa đầy đủ",
            'retry_count': retry_count
        }
