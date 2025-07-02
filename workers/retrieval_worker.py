from tools_and_services import VectorSearchTool, WebSearchTool, RerankTool
from tools_and_services.llm_services import LLMService
from typing import List, Dict
import asyncio
import logging

class RetrievalWorker:
    def __init__(self):
        self.vector_search_tool = VectorSearchTool()
        self.web_search_tool = WebSearchTool()
        self.rerank_tool = RerankTool()
        self.llm_service = LLMService()
        self.logger = logging.getLogger(__name__)
    
    async def retrieve_vector_chunks(self, structured_query: str) -> List[Dict]:
        """Retrieve chunks from vector database"""
        try:
            chunks = await self.vector_search_tool.search_knowledge_base(structured_query)
            self.logger.info(f"Retrieved {len(chunks)} chunks from vector database")
            return chunks
        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {e}")
            return []
    
    async def retrieve_web_content(self, aug_queries: List[str]) -> str:
        """Retrieve and summarize web content"""
        try:
            if not aug_queries:
                return ""
            
            # Search and fetch web content
            query_contents = await self.web_search_tool.search_and_fetch(aug_queries)
            
            if not query_contents:
                return ""
            
            # Summarize content for each augmented query
            summaries = []
            for aug_query, contents in query_contents.items():
                if contents:
                    # Merge content from all URLs for this query
                    merged_content = '\n\n'.join([item['content'] for item in contents])
                    
                    # Summarize using LLM
                    summary = await self.llm_service.summarize_web_content(aug_query, merged_content)
                    if summary:
                        summaries.append(f"Về câu hỏi '{aug_query}':\n{summary}")
            
            web_results = '\n\n'.join(summaries)
            self.logger.info(f"Generated web summaries for {len(summaries)} queries")
            return web_results
            
        except Exception as e:
            self.logger.error(f"Web content retrieval failed: {e}")
            return ""
    
    async def rerank_and_merge_context(self, structured_query: str, chunks: List[Dict], web_results: str) -> str:
        """Rerank chunks and merge with web results"""
        try:
            # Rerank chunks
            if chunks:
                reranked_chunks = await self.rerank_tool.get_top_k_reranked(
                    structured_query, chunks, top_k=5
                )
                
                # Format chunks
                chunk_texts = []
                for chunk in reranked_chunks:
                    content = chunk.get('content', '')
                    score = chunk.get('rerank_score', 0)
                    chunk_texts.append(f"[Score: {score:.3f}] {content}")
                
                chunks_context = '\n\n'.join(chunk_texts)
            else:
                chunks_context = ""
            
            # Combine chunks and web results
            context_parts = []
            if chunks_context:
                context_parts.append(f"=== THÔNG TIN TỪ CƠ SỞ KIẾN THỨC ===\n{chunks_context}")
            if web_results:
                context_parts.append(f"=== THÔNG TIN TỪ WEB ===\n{web_results}")
            
            context = '\n\n'.join(context_parts)
            self.logger.info(f"Merged context with {len(chunks)} chunks and web results")
            return context
            
        except Exception as e:
            self.logger.error(f"Context merging failed: {e}")
            return web_results  # Fallback to web results only
    
    async def retrieve_context(self, structured_query: str, aug_queries: List[str]) -> str:
        """Main method to retrieve and merge all context"""
        try:
            # Run vector search and web search in parallel
            vector_task = self.retrieve_vector_chunks(structured_query)
            web_task = self.retrieve_web_content(aug_queries)
            
            chunks, web_results = await asyncio.gather(vector_task, web_task)
            
            # Rerank and merge context
            context = await self.rerank_and_merge_context(structured_query, chunks, web_results)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Context retrieval failed: {e}")
            return ""
