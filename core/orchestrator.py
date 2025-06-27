from .workflows import IndexingWorkflow, QAWorkflow
from typing import Dict, Any, Optional, AsyncGenerator
import logging

class Orchestrator:
    """Main orchestrator using LangGraph to coordinate workflows"""
    
    def __init__(self):
        self.indexing_workflow = IndexingWorkflow()
        self.qa_workflow = QAWorkflow()
        self.logger = logging.getLogger(__name__)
    
    async def run_indexing(self, csv_file_path: str, callback=None) -> Dict[str, Any]:
        """Execute indexing workflow"""
        try:
            self.logger.info(f"Orchestrator: Starting indexing for {csv_file_path}")
            result = await self.indexing_workflow.run(csv_file_path, callback)
            self.logger.info(f"Orchestrator: Indexing completed with success={result['success']}")
            return result
        except Exception as e:
            self.logger.error(f"Orchestrator: Indexing failed - {e}")
            return {
                'success': False,
                'message': f'Orchestrator indexing error: {str(e)}',
                'document_count': 0
            }
    
    async def run_qa(self, query: str, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Execute Q&A workflow"""
        try:
            self.logger.info(f"Orchestrator: Starting Q&A for user {user_id}")
            result = await self.qa_workflow.run(query, user_id, conversation_id)
            self.logger.info(f"Orchestrator: Q&A completed with success={result['success']}")
            return result
        except Exception as e:
            self.logger.error(f"Orchestrator: Q&A failed - {e}")
            return {
                'success': False,
                'answer': 'Xin lỗi, hệ thống gặp sự cố. Vui lòng thử lại sau.',
                'error': str(e)
            }
    
    async def run_qa_streaming(self, query: str, user_id: str, conversation_id: str) -> AsyncGenerator[str, None]:
        """Execute Q&A workflow with streaming response"""
        try:
            self.logger.info(f"Orchestrator: Starting streaming Q&A for user {user_id}")
            
            # For now, run the full workflow first, then simulate streaming
            result = await self.qa_workflow.run(query, user_id, conversation_id)
            
            if result['success']:
                answer = result['answer']
                # Simulate streaming by yielding chunks
                words = answer.split()
                chunk_size = 3
                
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if i + chunk_size < len(words):
                        chunk += ' '
                    yield chunk
            else:
                yield result['answer']
                
        except Exception as e:
            self.logger.error(f"Orchestrator: Streaming Q&A failed - {e}")
            yield "Xin lỗi, hệ thống gặp sự cố. Vui lòng thử lại sau."
    
    def get_workflow_status(self) -> Dict[str, Dict[str, str]]:
        """Get status of all workflows"""
        return {
            'indexing': self.indexing_workflow.get_status(),
            'qa': self.qa_workflow.get_status(),
            'orchestrator': {
                'status': 'ready',
                'description': 'Main orchestrator coordinating all workflows'
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        try:
            status = self.get_workflow_status()
            return {
                'healthy': True,
                'message': 'All workflows operational',
                'workflows': status
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'message': f'Health check failed: {str(e)}',
                'workflows': {}
            }
    
    async def close(self):
        """Close all workflow connections"""
        try:
            await self.qa_workflow.close()
            self.logger.info("Orchestrator: All connections closed")
        except Exception as e:
            self.logger.error(f"Error closing orchestrator: {e}")
