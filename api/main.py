from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Literal
import uvicorn
import os
import sys
from loguru import logger

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import workflows directly
from agent import IndexingWorkflow, MedicalWorkflow

# Initialize FastAPI app
app = FastAPI(
    title="Drug Agentic Chatbot API",
    description="API for drug information chatbot with indexing and medical query capabilities",
    version="1.0.0"
)

# Initialize workflows
indexing_workflow = IndexingWorkflow()
medical_workflow = MedicalWorkflow()


# Pydantic models for request/response
class IndexingRequest(BaseModel):
    index_type: Literal["intent", "knowledge"]
    csv_file_path: str

class IndexingResponse(BaseModel):
    success: bool
    message: str
    document_count: int
    stats: Dict[str, Any]

class MedicalRequest(BaseModel):
    query: str
    user_id: str
    conversation_id: str

class MedicalResponse(BaseModel):
    intent: str
    answer: str

class HealthResponse(BaseModel):
    status: str
    service: str
    message: str


# Indexing endpoints
@app.get("/indexing/health", response_model=HealthResponse)
async def indexing_health_check():
    """Health check endpoint for indexing service"""
    try:
        return HealthResponse(
            status="healthy",
            service="indexing",
            message="Indexing workflow is running"
        )
    except Exception as e:
        logger.error(f"Indexing health check failed: {e}")
        raise HTTPException(status_code=500, detail="Indexing service unavailable")


@app.post("/indexing/run", response_model=IndexingResponse)
async def run_indexing(request: IndexingRequest):
    """Run indexing workflow for intent or knowledge data"""
    try:
        logger.info(f"Starting indexing: {request.index_type} from {request.csv_file_path}")
        
        result = await indexing_workflow.run(
            index_type=request.index_type,
            csv_file_path=request.csv_file_path
        )
        
        return IndexingResponse(
            success=result["success"],
            message=result["message"],
            document_count=result["document_count"],
            stats=result["stats"]
        )
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Indexing failed: {str(e)}"
        )


# Medical endpoints
@app.get("/medical/health", response_model=HealthResponse)
async def medical_health_check():
    """Health check endpoint for medical service"""
    try:
        return HealthResponse(
            status="healthy",
            service="medical",
            message="Medical workflow is running"
        )
    except Exception as e:
        logger.error(f"Medical health check failed: {e}")
        raise HTTPException(status_code=500, detail="Medical service unavailable")


@app.post("/medical/run", response_model=MedicalResponse)
async def run_medical_query(request: MedicalRequest):
    """Run medical workflow to process drug-related queries"""
    try:
        logger.info(f"Processing medical query for user {request.user_id}: {request.query}")
        
        result = await medical_workflow.run(
            query=request.query,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        return MedicalResponse(
            intent=result["intent"],
            answer=result["answer_text"]
        )
        
    except Exception as e:
        logger.error(f"Medical query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Medical query failed: {str(e)}"
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Drug Agentic Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "indexing": {
                "health": "/indexing/health",
                "run": "/indexing/run"
            },
            "medical": {
                "health": "/medical/health", 
                "run": "/medical/run"
            }
        }
    }


# Main function to run the server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )