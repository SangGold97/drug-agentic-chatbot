from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import uvicorn
from loguru import logger
import sys
import os

# Add parent directory to path to import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools_and_services import EmbeddingTool, RerankTool, VectorDBTool, WebSearchTool
from tools_and_services.llm_services.llm_services import LLMService

# Initialize FastAPI app
app = FastAPI(
    title="Drug Agentic Chatbot Tools API",
    description="API endpoints for embedding, rerank, vector DB, web search, and LLM tools",
    version="1.0.0"
)

# Global tool instances
embedding_tool = None
rerank_tool = None
vector_db_tool = None
web_search_tool = None
llm_services = {}

# Request models
class EmbeddingRequest(BaseModel):
    texts: List[str]

class RerankRequest(BaseModel):
    query: str
    chunks: List[Dict]

class VectorSearchRequest(BaseModel):
    query_embedding: List[float]
    collection_name: str

class VectorInsertRequest(BaseModel):
    collection_name: str
    documents: List[Dict[str, Any]]

class VectorDBDeleteRequest(BaseModel):
    collection_name: str

class WebSearchRequest(BaseModel):
    structured_queries: List[str]

class LLMRequest(BaseModel):
    service_name: str
    query: Optional[str] = None
    structured_query: Optional[str] = None
    context: Optional[str] = None
    original_query: Optional[str] = None
    chat_history: Optional[List[Dict]] = None

# Response models
class HealthResponse(BaseModel):
    status: str
    message: str

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

class DimensionResponse(BaseModel):
    dimension: int

class RerankResponse(BaseModel):
    reranked_chunks: List[Dict]

class VectorSearchResponse(BaseModel):
    results: List[Dict]

class VectorInsertResponse(BaseModel):
    status: str
    message: str

class VectorDBStatsResponse(BaseModel):
    status: str
    message: str
    stats: Dict[str, Any]

class VectorDBDeleteResponse(BaseModel):
    status: str
    message: str

class WebSearchResponse(BaseModel):
    results: Dict[str, List[Dict]]

class LLMResponse(BaseModel):
    response: str

@app.on_event("startup")
async def startup_event():
    """Initialize all tools on startup"""
    global embedding_tool, rerank_tool, vector_db_tool, web_search_tool, llm_services
    
    logger.info("Initializing tools...")
    
    # Initialize tools
    embedding_tool = EmbeddingTool()
    rerank_tool = RerankTool()
    vector_db_tool = VectorDBTool()
    web_search_tool = WebSearchTool()
    
    # Initialize LLM services
    service_names = ['answer']
    for service_name in service_names:
        try:
            llm_services[service_name] = LLMService(service_name)
            logger.info(f"LLM service '{service_name}' initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service '{service_name}': {e}")
    
    # Load models on GPU
    logger.info("Loading embedding model...")
    embedding_tool.load_model()
    
    logger.info("Loading rerank model...")
    rerank_tool.load_model()

    logger.info("Connecting to vector database...")
    await vector_db_tool.connect()
    
    logger.info("All tools initialized successfully!")

# Embedding endpoints
@app.post("/embedding/generate_embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    """Generate embeddings for input texts"""
    try:
        if embedding_tool is None:
            raise HTTPException(status_code=503, detail="Embedding tool not initialized")
        
        embeddings = await embedding_tool.generate_embedding(request.texts)
        return EmbeddingResponse(embeddings=embeddings)
    
    except Exception as e:
        logger.error(f"Error in generate_embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embedding/get_embedding_dimension", response_model=DimensionResponse)
async def get_embedding_dimension():
    """Get embedding dimension"""
    try:
        if embedding_tool is None:
            raise HTTPException(status_code=503, detail="Embedding tool not initialized")
        
        dimension = embedding_tool.get_embedding_dimension()
        return DimensionResponse(dimension=dimension)
    
    except Exception as e:
        logger.error(f"Error in get_embedding_dimension: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embedding/health_check", response_model=HealthResponse)
async def embedding_health_check():
    """Health check for embedding service"""
    try:
        if embedding_tool is None:
            return HealthResponse(status="error", message="Embedding tool not initialized")
        
        health_status = embedding_tool.health_check()
        return HealthResponse(**health_status)
    
    except Exception as e:
        logger.error(f"Error in embedding health check: {e}")
        return HealthResponse(status="error", message=str(e))

# Rerank endpoints
@app.post("/rerank/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """Rerank chunks based on query relevance"""
    try:
        if rerank_tool is None:
            raise HTTPException(status_code=503, detail="Rerank tool not initialized")
        
        reranked_chunks = await rerank_tool.rerank(
            query=request.query,
            chunks=request.chunks
        )
        return RerankResponse(reranked_chunks=reranked_chunks)
    
    except Exception as e:
        logger.error(f"Error in rerank: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rerank/health_check", response_model=HealthResponse)
async def rerank_health_check():
    """Health check for rerank service"""
    try:
        if rerank_tool is None:
            return HealthResponse(status="error", message="Rerank tool not initialized")
        
        health_status = rerank_tool.health_check()
        return HealthResponse(**health_status)
    
    except Exception as e:
        logger.error(f"Error in rerank health check: {e}")
        return HealthResponse(status="error", message=str(e))

# Vector search endpoints
@app.post("/vector_db/search", response_model=VectorSearchResponse)
async def search(request: VectorSearchRequest):
    """Search vector database using vector similarity"""
    try:
        if vector_db_tool is None:
            raise HTTPException(status_code=503, detail="Vector DB tool not initialized")

        results = await vector_db_tool.search(query_embedding=request.query_embedding,
                                               collection_name=request.collection_name)
        return VectorSearchResponse(results=results)
    
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/vector_db/insert", response_model=VectorInsertResponse)
async def insert(request: VectorInsertRequest):
    """Insert documents into vector database"""
    try:
        if vector_db_tool is None:
            raise HTTPException(status_code=503, detail="Vector DB tool not initialized")

        result = await vector_db_tool.insert(collection_name=request.collection_name,
                                             documents=request.documents)
        return VectorInsertResponse(**result)

    except Exception as e:
        logger.error(f"Error in vector insert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector_db/stats")
async def get_vector_db_stats():
    """Get statistics for all vector database collections"""
    try:
        if vector_db_tool is None:
            raise HTTPException(status_code=503, detail="Vector DB tool not initialized")

        result = vector_db_tool.get_stats()
        return VectorDBStatsResponse(**result)

    except Exception as e:
        logger.error(f"Error in vector DB stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector_db/delete_collection", response_model=VectorDBDeleteResponse)
async def delete_collection(request: VectorDBDeleteRequest):
    """Delete a collection from vector database"""
    try:
        if vector_db_tool is None:
            raise HTTPException(status_code=503, detail="Vector DB tool not initialized")

        result = vector_db_tool.delete_collection(collection_name=request.collection_name)
        return VectorDBDeleteResponse(**result)

    except Exception as e:
        logger.error(f"Error in vector DB delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/vector_db/health_check", response_model=HealthResponse)
async def vector_db_health_check():
    """Health check for vector DB service"""
    try:
        if vector_db_tool is None:
            return HealthResponse(status="error", message="Vector DB tool not initialized")

        health_status = vector_db_tool.health_check()
        return HealthResponse(**health_status)
    
    except Exception as e:
        logger.error(f"Error in vector DB health check: {e}")
        return HealthResponse(status="error", message=str(e))

# Web search endpoints
@app.post("/web_search/search_and_fetch", response_model=WebSearchResponse)
async def search_and_fetch(request: WebSearchRequest):
    """Search and fetch content for multiple structured queries"""
    try:
        if web_search_tool is None:
            raise HTTPException(status_code=503, detail="Web search tool not initialized")
        
        results = await web_search_tool.search_and_fetch(request.structured_queries)
        return WebSearchResponse(results=results)
    
    except Exception as e:
        logger.error(f"Error in search_and_fetch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/web_search/health_check", response_model=HealthResponse)
async def web_search_health_check():
    """Health check for web search service"""
    try:
        if web_search_tool is None:
            return HealthResponse(status="error", message="Web search tool not initialized")
        
        health_status = web_search_tool.health_check()
        return HealthResponse(**health_status)
    
    except Exception as e:
        logger.error(f"Error in web search health check: {e}")
        return HealthResponse(status="error", message=str(e))

# LLM endpoints
@app.post("/llm/generate_response", response_model=LLMResponse)
async def generate_response(request: LLMRequest):
    """Generate response using LLM service"""
    try:
        # if request.service_name not in llm_services:
        #     raise HTTPException(status_code=400, detail=f"Invalid service name: {request.service_name}")
        
        service = llm_services['answer']
        
        # Prepare arguments based on service type
        if request.service_name == 'structured_query_generator':
            if not request.original_query:
                raise HTTPException(status_code=400, detail="original_query is required for structured_query_generator")
            response = await service.generate_response(request.original_query)
        
        elif request.service_name == 'reflection':
            if not request.structured_query or not request.context:
                raise HTTPException(status_code=400, detail="structured_query and context are required for reflection")
            response = await service.generate_response(request.structured_query, request.context)
        
        elif request.service_name == 'general':
            if not request.query:
                raise HTTPException(status_code=400, detail="query is required for general")
            response = await service.generate_response(request.query)
        
        elif request.service_name == 'answer':
            if not request.original_query or not request.context:
                raise HTTPException(status_code=400, detail="original_query and context are required for answer")
            chat_history = request.chat_history or []
            response = await service.generate_response(request.original_query, request.context, chat_history)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown service: {request.service_name}")
        
        return LLMResponse(response=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm/health_check", response_model=Dict[str, HealthResponse])
async def llm_health_check():
    """Health check for LLM services"""
    health_status = {}
    
    for service_name, service in llm_services.items():
        try:
            if service and service.model and service.processor:
                health_status[service_name] = HealthResponse(status="healthy", message="Service is running")
            else:
                health_status[service_name] = HealthResponse(status="error", message="Service not properly initialized")
        except Exception as e:
            health_status[service_name] = HealthResponse(status="error", message=str(e))
    
    return health_status

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Drug Agentic Chatbot Tools API",
        "version": "1.0.0",
        "services": [
            "embedding",
            "rerank", 
            "vector_db",
            "web_search",
            "llm"
        ],
        "port": 8001
    }

# Global health check
@app.get("/health", response_model=Dict[str, HealthResponse])
async def global_health_check():
    """Global health check for all services"""
    health_status = {}
    
    try:
        # Check embedding service
        if embedding_tool:
            health_status["embedding"] = HealthResponse(**embedding_tool.health_check())
        else:
            health_status["embedding"] = HealthResponse(status="error", message="Not initialized")
        
        # Check rerank service
        if rerank_tool:
            health_status["rerank"] = HealthResponse(**rerank_tool.health_check())
        else:
            health_status["rerank"] = HealthResponse(status="error", message="Not initialized")

        # Check vector DB service
        if vector_db_tool:
            health_status["vector_db"] = HealthResponse(**vector_db_tool.health_check())
        else:
            health_status["vector_db"] = HealthResponse(status="error", message="Not initialized")

        # Check web search service
        if web_search_tool:
            health_status["web_search"] = HealthResponse(**web_search_tool.health_check())
        else:
            health_status["web_search"] = HealthResponse(status="error", message="Not initialized")

        # Check LLM services
        llm_health = {}
        for service_name, service in llm_services.items():
            try:
                if service and service.model and service.processor:
                    llm_health[service_name] = HealthResponse(status="healthy", message="Service is running")
                else:
                    llm_health[service_name] = HealthResponse(status="error", message="Service not properly initialized")
            except Exception as e:
                llm_health[service_name] = HealthResponse(status="error", message=str(e))
        health_status["llm"] = llm_health
            
    except Exception as e:
        logger.error(f"Error in global health check: {e}")
    
    return health_status

if __name__ == "__main__":
    logger.info("Starting Drug Agentic Chatbot Tools API on port 8001...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
