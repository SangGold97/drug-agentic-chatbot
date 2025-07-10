import os
import asyncio
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from loguru import logger

# Import the indexing workers
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from workers.index_intent import IndexIntent
from workers.index_knowledge import IndexKnowledge


class IndexingState(TypedDict):
    """State for the indexing workflow"""
    index_type: str
    csv_file_path: str
    success: bool
    message: str
    document_count: int
    stats: Dict[str, Any]


class IndexingWorkflow:
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize IndexingWorkflow with base URL for the tools API
        
        Args:
            base_url: Base URL for the tools and services API
        """
        self.base_url = base_url
        self.intent_indexer = IndexIntent(base_url)
        self.knowledge_indexer = IndexKnowledge(base_url)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(IndexingState)
        
        # Add nodes
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("index_intent", self._index_intent)
        workflow.add_node("index_knowledge", self._index_knowledge)
        workflow.add_node("get_stats", self._get_stats)
        
        # Define the workflow
        workflow.set_entry_point("validate_input")
        
        # Conditional routing based on index_type
        workflow.add_conditional_edges(
            "validate_input",
            self._route_indexing,
            {
                "intent": "index_intent",
                "knowledge": "index_knowledge",
                "error": END
            }
        )
        
        # Both indexing paths lead to getting stats
        workflow.add_edge("index_intent", "get_stats")
        workflow.add_edge("index_knowledge", "get_stats")
        workflow.add_edge("get_stats", END)
        
        return workflow.compile()

    async def _validate_input(self, state: IndexingState) -> IndexingState:
        """Validate input parameters"""
        logger.info(f"Validating input: index_type={state['index_type']}, csv_file_path={state['csv_file_path']}")
        
        # Check index_type
        if state["index_type"] not in ["intent", "knowledge"]:
            state["success"] = False
            state["message"] = f"Invalid index_type: {state['index_type']}. Must be 'intent' or 'knowledge'"
            return state
        
        # Check if CSV file exists
        if not os.path.exists(state["csv_file_path"]):
            state["success"] = False
            state["message"] = f"CSV file not found: {state['csv_file_path']}"
            return state
        
        logger.info("Input validation passed")
        return state

    def _route_indexing(self, state: IndexingState) -> str:
        """Route to appropriate indexing node based on index_type"""
        if not state.get("success", True):
            return "error"
        
        if state["index_type"] == "intent":
            return "intent"
        elif state["index_type"] == "knowledge":
            return "knowledge"
        else:
            return "error"

    async def _index_intent(self, state: IndexingState) -> IndexingState:
        """Index intent queries"""
        logger.info("Starting intent indexing...")
        
        try:
            result = await self.intent_indexer.run(state["csv_file_path"])
            
            state["success"] = result["success"]
            state["message"] = result["message"]
            if "document_count" in result:
                state["document_count"] = result["document_count"]
            
            logger.info(f"Intent indexing completed: {state['success']}")
            
        except Exception as e:
            logger.error(f"Intent indexing failed: {e}")
            state["success"] = False
            state["message"] = f"Intent indexing failed: {str(e)}"
        
        return state

    async def _index_knowledge(self, state: IndexingState) -> IndexingState:
        """Index knowledge base"""
        logger.info("Starting knowledge indexing...")
        
        try:
            result = await self.knowledge_indexer.run(state["csv_file_path"])
            
            state["success"] = result["success"]
            state["message"] = result["message"]
            if "document_count" in result:
                state["document_count"] = result["document_count"]
            
            logger.info(f"Knowledge indexing completed: {state['success']}")
            
        except Exception as e:
            logger.error(f"Knowledge indexing failed: {e}")
            state["success"] = False
            state["message"] = f"Knowledge indexing failed: {str(e)}"
        
        return state

    async def _get_stats(self, state: IndexingState) -> IndexingState:
        """Get collection statistics"""
        logger.info("Getting collection statistics...")
        
        try:
            if state["index_type"] == "intent":
                stats_result = await self.intent_indexer.get_stats_collection()
            else:  # knowledge
                stats_result = await self.knowledge_indexer.get_stats_collection()
            
            if stats_result["success"]:
                state["stats"] = stats_result["stats"]
                logger.info(f"Collection stats: {state['stats']}")
            else:
                logger.warning("Failed to get collection stats")
                state["stats"] = {}
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            state["stats"] = {}
        
        return state

    async def run(self, index_type: Literal["intent", "knowledge"], csv_file_path: str) -> Dict[str, Any]:
        """
        Run the indexing workflow
        
        Args:
            index_type: Type of indexing ('intent' or 'knowledge')
            csv_file_path: Path to the CSV file to index
            
        Returns:
            Dict containing success status, message, document count, and stats
        """
        logger.info(f"Starting indexing workflow: {index_type} from {csv_file_path}")
        
        # Initialize state
        initial_state = IndexingState(
            index_type=index_type,
            csv_file_path=csv_file_path,
            success=True,
            message="",
            document_count=0,
            stats={}
        )
        
        try:
            # Run the workflow
            result = await self.graph.ainvoke(initial_state)
            
            return {
                "success": result["success"],
                "message": result["message"],
                "document_count": result.get("document_count", 0),
                "stats": result.get("stats", {})
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "message": f"Workflow execution failed: {str(e)}",
                "document_count": 0,
                "stats": {}
            }

    def health_check(self) -> Dict[str, str]:
        """Health check for indexing workflow"""
        try:
            return {"status": "healthy", "message": "Indexing workflow is ready"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}

async def main():
    """Test function for the IndexingWorkflow class"""
    # Initialize workflow
    workflow = IndexingWorkflow()
    
    logger.info("Testing IndexingWorkflow...")
    
    # Test cases
    test_cases = [
        {
            "index_type": "intent",
            "csv_file_path": "../../intent_queries/intent_queries.csv",
            "description": "Intent queries indexing"
        },
        {
            "index_type": "knowledge", 
            "csv_file_path": "../../knowledge_base/knowledge_base.csv",
            "description": "Knowledge base indexing"
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {test_case['description']}")
        logger.info(f"{'='*50}")
        
        try:
            # Run workflow
            result = await workflow.run(
                index_type=test_case["index_type"],
                csv_file_path=test_case["csv_file_path"]
            )
            
            # Display results
            logger.info(f"Success: {result['success']}")
            logger.info(f"Message: {result['message']}")
            logger.info(f"Document count: {result['document_count']}")
            logger.info("Collection stats:")
            for collection, count in result['stats'].items():
                logger.info(f"  - {collection}: {count} documents")
                
        except Exception as e:
            logger.error(f"Test failed: {e}")
    
    logger.info("\nIndexing workflow testing completed!")


if __name__ == "__main__":
    asyncio.run(main())