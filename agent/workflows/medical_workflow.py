from typing import Dict, Any, TypedDict, Annotated, Literal
import asyncio
from loguru import logger
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Import workers
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from workers.intent_classification import IntentClassification
from workers.structured_query_generator import StructuredQueryGenerator
from workers.retriever import Retriever
from workers.reflection import Reflection
from workers.answer import Answer
from workers.save_conversation import SaveConversation


class MedicalWorkflowState(TypedDict):
    """State schema for the medical workflow"""
    # Input data
    query: str
    user_id: str
    conversation_id: str
    
    # Intermediate processing data
    intent: str
    structured_query: str
    retriever_results: Dict[str, Any]
    sufficient: bool
    follow_up_query: str
    second_retrieval_results: Dict[str, Any]
    combined_results: Dict[str, Any]
    
    # Output data
    answer_text: str
    save_result: Dict[str, Any]


class MedicalWorkflow:
    """Medical workflow using LangGraph for agentic RAG chatbot"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize MedicalWorkflow with workers
        
        Args:
            base_url: Base URL for the tools and services API
        """
        self.base_url = base_url
        
        # Initialize workers
        self.intent_classifier = IntentClassification(base_url)
        self.query_generator = StructuredQueryGenerator(base_url)
        self.retriever = Retriever(base_url)
        self.reflection = Reflection(base_url)
        self.answer_worker = Answer(base_url)
        self.save_conversation = SaveConversation(base_url)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create workflow graph
        workflow = StateGraph(MedicalWorkflowState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("generate_structured_query", self._generate_structured_query_node)
        workflow.add_node("retrieve_information", self._retrieve_information_node)
        workflow.add_node("check_reflection", self._check_reflection_node)
        workflow.add_node("retrieve_more_information", self._retrieve_more_information_node)
        workflow.add_node("generate_general_answer", self._generate_general_answer_node)
        workflow.add_node("generate_medical_answer", self._generate_medical_answer_node)
        workflow.add_node("save_conversation", self._save_conversation_node)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "general": "generate_general_answer",
                "medical": "generate_structured_query"
            }
        )
        
        workflow.add_edge("generate_structured_query", "retrieve_information")
        workflow.add_edge("retrieve_information", "check_reflection")
        
        workflow.add_conditional_edges(
            "check_reflection",
            self._check_if_sufficient,
            {
                "sufficient": "generate_medical_answer",
                "insufficient": "retrieve_more_information"
            }
        )
        
        workflow.add_edge("retrieve_more_information", "generate_medical_answer")
        workflow.add_edge("generate_general_answer", "save_conversation")
        workflow.add_edge("generate_medical_answer", "save_conversation")
        workflow.add_edge("save_conversation", END)
        
        return workflow.compile()
    
    # Node functions
    async def _classify_intent_node(self, state: MedicalWorkflowState) -> Dict[str, Any]:
        """Classify intent of user query"""
        logger.info(f"Classifying intent for query: {state['query']}")
        intent = await self.intent_classifier.run(state["query"])
        logger.info(f"Intent classified as: {intent}")
        return {"intent": intent}
    
    async def _generate_structured_query_node(self, state: MedicalWorkflowState) -> Dict[str, Any]:
        """Generate structured query from user input"""
        logger.info("Generating structured query")
        structured_query = await self.query_generator.run(state["query"])
        logger.info(f"Structured query generated: {structured_query}")
        return {"structured_query": structured_query}
    
    async def _retrieve_information_node(self, state: MedicalWorkflowState) -> Dict[str, Any]:
        """Retrieve information using the structured query"""
        logger.info("Retrieving information with web_search=True, vector_search=True")
        results = await self.retriever.run(
            state["structured_query"], 
            web_search=True, 
            vector_search=True
        )
        logger.info(f"Retrieved {len(results)} result types")
        return {"retriever_results": results}
    
    async def _check_reflection_node(self, state: MedicalWorkflowState) -> Dict[str, Any]:
        """Check if retrieved information is sufficient"""
        logger.info("Checking reflection")
        result = await self.reflection.run(
            state["structured_query"], 
            state["retriever_results"]
        )
        logger.info(f"Reflection result - sufficient: {result['sufficient']}")
        return {
            "sufficient": result["sufficient"],
            "follow_up_query": result["follow_up_query"]
        }
    
    async def _retrieve_more_information_node(self, state: MedicalWorkflowState) -> Dict[str, Any]:
        """Retrieve additional information using follow-up query"""
        logger.info("Retrieving more information with web_search=True, vector_search=False")
        results = await self.retriever.run(
            state["follow_up_query"], 
            web_search=True, 
            vector_search=False
        )
        
        # Combine results from both retrievals
        combined = {
            "web_search": {},
            "vector_search": state["retriever_results"].get("vector_search", [])
        }
        
        # Merge web search results
        web_search_1 = state["retriever_results"].get("web_search", {})
        web_search_2 = results.get("web_search", {})
        
        for query, results_list in web_search_1.items():
            combined["web_search"][query] = results_list
        
        for query, results_list in web_search_2.items():
            combined["web_search"][query] = results_list
        
        logger.info("Combined retrieval results")
        return {
            "second_retrieval_results": results,
            "combined_results": combined
        }
    
    async def _generate_general_answer_node(self, state: MedicalWorkflowState) -> Dict[str, Any]:
        """Generate general answer without context"""
        logger.info("Generating general answer")
        self.answer_worker.set_user_info(state["user_id"], state["conversation_id"])
        result = await self.answer_worker.run(
            query=state["query"],
            service_name="general"
        )
        logger.info("General answer generated")
        return {"answer_text": result["answer"]}
    
    async def _generate_medical_answer_node(self, state: MedicalWorkflowState) -> Dict[str, Any]:
        """Generate medical answer with context"""
        logger.info("Generating medical answer with context")
        self.answer_worker.set_user_info(state["user_id"], state["conversation_id"])
        
        # Use combined results if available, otherwise use retriever_results
        context = state.get("combined_results") or state["retriever_results"]
        
        result = await self.answer_worker.run(
            query=state["query"],
            service_name="answer",
            context=context
        )
        logger.info("Medical answer generated")
        return {"answer_text": result["answer"]}
    
    async def _save_conversation_node(self, state: MedicalWorkflowState) -> Dict[str, Any]:
        """Save conversation to database"""
        logger.info("Saving conversation")
        result = await self.save_conversation.run(
            user_id=state["user_id"],
            conversation_id=state["conversation_id"],
            query=state["query"],
            answer=state["answer_text"]
        )
        logger.info("Conversation saved")
        return {"save_result": result}
    
    # Edge routing functions
    def _route_by_intent(self, state: MedicalWorkflowState) -> Literal["general", "medical"]:
        """Route workflow based on intent classification"""
        return "general" if state["intent"] == "general" else "medical"
    
    def _check_if_sufficient(self, state: MedicalWorkflowState) -> Literal["sufficient", "insufficient"]:
        """Route based on sufficiency of retrieved information"""
        return "sufficient" if state["sufficient"] else "insufficient"
    
    async def run(self, query: str, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Run the medical workflow with the given inputs
        
        Args:
            query: User's query text
            user_id: User identifier
            conversation_id: Conversation identifier
            
        Returns:
            Dict containing the final state of the workflow
        """
        logger.info(f"Starting medical workflow for query: {query}")
        
        # Initialize state
        initial_state = MedicalWorkflowState(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            intent="",
            structured_query="",
            retriever_results={},
            sufficient=False,
            follow_up_query="",
            second_retrieval_results={},
            combined_results={},
            answer_text="",
            save_result={}
        )
        
        try:
            # Run the workflow
            result = await self.workflow.ainvoke(initial_state)
            logger.info("Medical workflow completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in medical workflow: {e}")
            raise e

    def health_check(self) -> Dict[str, str]:
        """Health check for medical workflow"""
        try:
            return {"status": "healthy", "message": "Medical workflow is ready"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}

async def main():
    """Test function for MedicalWorkflow"""
    logger.info("Testing MedicalWorkflow...")
    
    # Initialize workflow
    workflow = MedicalWorkflow()
    
    # Test cases
    test_cases = [
        {
            "query": "Thuốc meloxicam có tác dụng gì? Trong báo cáo PGx, tôi nên uống meloxicam không?",
            "user_id": "user_123",
            "conversation_id": "conv_456",
            "expected_intent": "medical"
        },
        {
            "query": "Thuốc codein có tác dụng gì, liều dùng như thế nào, có tác dụng phụ không? Kiểu gen của tôi có nên uống codein không?",
            "user_id": "user_123", 
            "conversation_id": "conv_456",
            "expected_intent": "medical"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n=== Test case {i} ===")
        logger.info(f"Query: {test_case['query']}")
        
        try:
            result = await workflow.run(
                query=test_case["query"],
                user_id=test_case["user_id"],
                conversation_id=test_case["conversation_id"]
            )
            
            logger.info(f"Intent: {result.get('intent')}")
            logger.info(f"Answer: {result.get('answer_text')}")
            logger.info(f"Save result: {result.get('save_result')}")
            
        except Exception as e:
            logger.error(f"Test case {i} failed: {e}")
    
    logger.info("Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())