from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import uuid
from datetime import datetime
import json

# Import orchestrator
from core import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Drug Agentic Chatbot API",
    description="AI-powered chatbot for drug and medical information with RAG and multi-agent architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = Orchestrator()

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    user_id: str = Field(..., description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation ID (auto-generated if not provided)")

class IndexingRequest(BaseModel):
    csv_file_path: str = Field(..., description="Path to CSV file for indexing")

class QueryResponse(BaseModel):
    success: bool
    answer: str
    conversation_id: str
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class IndexingResponse(BaseModel):
    success: bool
    message: str
    document_count: int
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    healthy: bool
    message: str
    timestamp: str
    workflows: Dict[str, Dict[str, str]]

# Endpoints
@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Drug Agentic Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "qa": "/qa",
            "qa_stream": "/qa/stream",
            "index": "/index",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/qa", response_model=QueryResponse, summary="Question & Answer")
async def query_answer(request: QueryRequest):
    """
    Process user query and return answer using multi-agent RAG system
    
    - **query**: User's question about drugs, medical conditions, etc.
    - **user_id**: Unique identifier for the user
    - **conversation_id**: Optional conversation ID (auto-generated if not provided)
    """
    start_time = datetime.now()
    
    try:
        # Generate conversation_id if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        logger.info(f"Processing Q&A request from user {request.user_id}")
        
        # Execute Q&A workflow
        result = await orchestrator.run_qa(
            query=request.query,
            user_id=request.user_id,
            conversation_id=conversation_id
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if result['success']:
            return QueryResponse(
                success=True,
                answer=result['answer'],
                conversation_id=conversation_id,
                processing_time=processing_time,
                metadata={
                    "final_state": result.get('final_state', {}).get('current_step'),
                    "query_length": len(request.query)
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Q&A processing failed: {result.get('error', 'Unknown error')}"
            )
    
    except Exception as e:
        logger.error(f"Q&A endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa/stream", summary="Streaming Question & Answer")
async def query_answer_stream(request: QueryRequest):
    """
    Process user query and return streaming answer
    
    Returns Server-Sent Events (SSE) stream for real-time response
    """
    try:
        # Generate conversation_id if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        logger.info(f"Processing streaming Q&A request from user {request.user_id}")
        
        def generate_stream():
            try:
                # Send initial metadata
                yield f"data: {json.dumps({'type': 'metadata', 'conversation_id': conversation_id})}\n\n"
                
                # Stream answer chunks
                async def async_gen():
                    async for chunk in orchestrator.run_qa_streaming(
                        query=request.query,
                        user_id=request.user_id,
                        conversation_id=conversation_id
                    ):
                        yield chunk
                
                # Convert async generator to sync for FastAPI
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    async_iterator = async_gen()
                    while True:
                        try:
                            chunk = loop.run_until_complete(async_iterator.__anext__())
                            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                        except StopAsyncIteration:
                            break
                finally:
                    loop.close()
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    
    except Exception as e:
        logger.error(f"Streaming Q&A endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", response_model=IndexingResponse, summary="Index Knowledge Base")
async def index_knowledge_base(request: IndexingRequest, background_tasks: BackgroundTasks):
    """
    Index CSV knowledge base into vector database
    
    - **csv_file_path**: Path to the CSV file containing knowledge base data
    
    This operation runs in the background for large datasets.
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting indexing for file: {request.csv_file_path}")
        
        # For large files, you might want to run this in background
        # For now, run synchronously
        result = await orchestrator.run_indexing(request.csv_file_path)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return IndexingResponse(
            success=result['success'],
            message=result['message'],
            document_count=result.get('document_count', 0),
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Indexing endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """
    Perform health check on all system components
    
    Returns status of orchestrator, workflows, and database connections
    """
    try:
        health_result = orchestrator.health_check()
        
        return HealthResponse(
            healthy=health_result['healthy'],
            message=health_result['message'],
            timestamp=datetime.now().isoformat(),
            workflows=health_result['workflows']
        )
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            healthy=False,
            message=f"Health check failed: {str(e)}",
            timestamp=datetime.now().isoformat(),
            workflows={}
        )

@app.get("/status", summary="System Status")
async def get_status():
    """Get detailed system status including loaded models and database connections"""
    try:
        workflow_status = orchestrator.get_workflow_status()
        
        return {
            "api_status": "running",
            "timestamp": datetime.now().isoformat(),
            "workflows": workflow_status,
            "system_info": {
                "api_version": "1.0.0",
                "uptime": "Available in production deployment"
            }
        }
    
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return {
            "api_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Background task for async indexing (if needed)
def background_indexing(csv_file_path: str):
    """Background task for large indexing operations"""
    try:
        logger.info(f"Background indexing started for: {csv_file_path}")
        
        # Run async function in background
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(orchestrator.run_indexing(csv_file_path))
            logger.info(f"Background indexing completed: {result['success']}")
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Background indexing failed: {e}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Drug Agentic Chatbot API starting up...")
    
    # Perform any initialization here
    # e.g., preload models, check database connections
    try:
        health = orchestrator.health_check()
        if health['healthy']:
            logger.info("✅ System startup successful - all components healthy")
        else:
            logger.warning(f"⚠️ System startup with warnings: {health['message']}")
    except Exception as e:
        logger.error(f"❌ System startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Drug Agentic Chatbot API shutting down...")
    
    try:
        await orchestrator.close()
        logger.info("✅ System shutdown completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "/qa", "/qa/stream", "/index", "/health", "/status", "/docs"
        ]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later.",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
