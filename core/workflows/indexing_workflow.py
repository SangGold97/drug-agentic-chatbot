from langchain.schema import BaseCallbackHandler
from workers import IndexingWorker
from typing import Dict, Any, Optional
import logging

class IndexingWorkflow:
    """Simple LangChain-based workflow for indexing knowledge base"""
    
    def __init__(self):
        self.indexing_worker = IndexingWorker()
        self.logger = logging.getLogger(__name__)
    
    async def run(self, csv_file_path: str, callback: Optional[BaseCallbackHandler] = None) -> Dict[str, Any]:
        """Execute indexing workflow"""
        try:
            self.logger.info(f"Starting indexing workflow for: {csv_file_path}")
            
            if callback:
                callback.on_text("🚀 Bắt đầu quá trình indexing knowledge base...\n")
            
            # Step 1: Index knowledge base
            if callback:
                callback.on_text("📚 Đọc và xử lý CSV file...\n")
            
            result = await self.indexing_worker.index_knowledge_base(csv_file_path)
            
            if result['success']:
                success_msg = f"✅ Indexing hoàn thành! Đã index {result.get('document_count', 0)} documents"
                self.logger.info(success_msg)
                if callback:
                    callback.on_text(f"{success_msg}\n")
            else:
                error_msg = f"❌ Indexing thất bại: {result['message']}"
                self.logger.error(error_msg)
                if callback:
                    callback.on_text(f"{error_msg}\n")
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Lỗi trong quá trình indexing: {str(e)}"
            self.logger.error(error_msg)
            if callback:
                callback.on_text(f"{error_msg}\n")
            
            return {
                'success': False,
                'message': str(e),
                'document_count': 0
            }
        finally:
            await self.indexing_worker.close()
    
    def get_status(self) -> Dict[str, str]:
        """Get workflow status"""
        return {
            'workflow_type': 'indexing',
            'status': 'ready',
            'description': 'Index CSV knowledge base into Milvus vector database'
        }
