from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from workers import (
    IntentClassificationWorker,
    QueryAugmentationWorker, 
    RetrievalWorker,
    ReflectionWorker,
    QAndAWorker,
    SaveConversationWorker
)
import asyncio
import logging

class QAState(TypedDict):
    # Input
    original_query: str
    user_id: str
    conversation_id: str
    
    # Intent classification
    intent: Optional[str]
    
    # Query augmentation
    structured_query: Optional[str]
    augmented_queries: Optional[List[str]]
    
    # Retrieval
    context: Optional[str]
    
    # Reflection
    reflection_result: Optional[Dict[str, Any]]
    final_context: Optional[str]
    
    # Answer generation
    answer: Optional[str]
    
    # Status tracking
    current_step: str
    is_complete: bool
    error_message: Optional[str]

class QAWorkflow:
    """LangGraph-based workflow for Q&A processing"""
    
    def __init__(self):
        # Initialize workers
        self.intent_worker = IntentClassificationWorker()
        self.augmentation_worker = QueryAugmentationWorker()
        self.retrieval_worker = RetrievalWorker()
        self.reflection_worker = ReflectionWorker()
        self.qa_worker = QAndAWorker()
        self.save_worker = SaveConversationWorker()
        
        self.logger = logging.getLogger(__name__)
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(QAState)
        
        # Add nodes
        workflow.add_node("intent_classification", self._intent_classification_node)
        workflow.add_node("query_augmentation", self._query_augmentation_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("reflection", self._reflection_node)
        workflow.add_node("answer_generation", self._answer_generation_node)
        workflow.add_node("save_conversation", self._save_conversation_node)
        workflow.add_node("general_response", self._general_response_node)
        
        # Set entry point
        workflow.set_entry_point("intent_classification")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "intent_classification",
            self._route_after_intent,
            {
                "medical": "query_augmentation",
                "general": "general_response"
            }
        )
        
        workflow.add_edge("query_augmentation", "retrieval")
        workflow.add_edge("retrieval", "reflection")
        
        workflow.add_conditional_edges(
            "reflection",
            self._route_after_reflection,
            {
                "sufficient": "answer_generation",
                "retry": "retrieval"
            }
        )
        
        workflow.add_edge("answer_generation", "save_conversation")
        workflow.add_edge("save_conversation", END)
        workflow.add_edge("general_response", END)
        
        return workflow.compile()
    
    async def _intent_classification_node(self, state: QAState) -> QAState:
        """Intent classification node"""
        try:
            self.logger.info("Executing intent classification")
            
            intent = await self.intent_worker.classify_intent(state["original_query"])
            
            return {
                **state,
                "intent": intent,
                "current_step": "intent_classification_complete"
            }
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return {
                **state,
                "intent": "medical",  # Default to medical
                "current_step": "intent_classification_error",
                "error_message": str(e)
            }
    
    async def _query_augmentation_node(self, state: QAState) -> QAState:
        """Query augmentation node"""
        try:
            self.logger.info("Executing query augmentation")
            
            result = await self.augmentation_worker.augment_query(state["original_query"])
            
            return {
                **state,
                "structured_query": result["structured_query"],
                "augmented_queries": result["augmented_queries"],
                "current_step": "query_augmentation_complete"
            }
        except Exception as e:
            self.logger.error(f"Query augmentation failed: {e}")
            return {
                **state,
                "structured_query": state["original_query"],
                "augmented_queries": [state["original_query"]],
                "current_step": "query_augmentation_error",
                "error_message": str(e)
            }
    
    async def _retrieval_node(self, state: QAState) -> QAState:
        """Retrieval node"""
        try:
            self.logger.info("Executing retrieval")
            
            context = await self.retrieval_worker.retrieve_context(
                state["structured_query"],
                state["augmented_queries"]
            )
            
            return {
                **state,
                "context": context,
                "current_step": "retrieval_complete"
            }
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return {
                **state,
                "context": "",
                "current_step": "retrieval_error",
                "error_message": str(e)
            }
    
    async def _reflection_node(self, state: QAState) -> QAState:
        """Reflection node"""
        try:
            self.logger.info("Executing reflection")
            
            reflection_result = await self.reflection_worker.iterative_reflection(
                state["structured_query"],
                state["augmented_queries"],
                state["context"]
            )
            
            return {
                **state,
                "reflection_result": reflection_result,
                "final_context": reflection_result["final_context"],
                "current_step": "reflection_complete"
            }
        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            return {
                **state,
                "reflection_result": {"sufficient": True, "retry_count": 0},
                "final_context": state["context"],
                "current_step": "reflection_error",
                "error_message": str(e)
            }
    
    async def _answer_generation_node(self, state: QAState) -> QAState:
        """Answer generation node"""
        try:
            self.logger.info("Executing answer generation")
            
            answer = await self.qa_worker.generate_answer(
                state["original_query"],
                state["final_context"],
                state["user_id"],
                state["conversation_id"]
            )
            
            return {
                **state,
                "answer": answer,
                "current_step": "answer_generation_complete"
            }
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return {
                **state,
                "answer": "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này.",
                "current_step": "answer_generation_error",
                "error_message": str(e)
            }
    
    async def _save_conversation_node(self, state: QAState) -> QAState:
        """Save conversation node"""
        try:
            self.logger.info("Executing save conversation")
            
            success = await self.save_worker.save_conversation(
                state["user_id"],
                state["conversation_id"],
                state["original_query"],
                state["answer"]
            )
            
            return {
                **state,
                "current_step": "save_conversation_complete",
                "is_complete": True
            }
        except Exception as e:
            self.logger.error(f"Save conversation failed: {e}")
            return {
                **state,
                "current_step": "save_conversation_error",
                "error_message": str(e),
                "is_complete": True  # Complete anyway
            }
    
    async def _general_response_node(self, state: QAState) -> QAState:
        """General response node for non-medical queries"""
        try:
            self.logger.info("Executing general response")
            
            answer = await self.qa_worker.generate_general_response(state["original_query"])
            
            return {
                **state,
                "answer": answer,
                "current_step": "general_response_complete",
                "is_complete": True
            }
        except Exception as e:
            self.logger.error(f"General response failed: {e}")
            return {
                **state,
                "answer": "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này.",
                "current_step": "general_response_error",
                "error_message": str(e),
                "is_complete": True
            }
    
    def _route_after_intent(self, state: QAState) -> str:
        """Route after intent classification"""
        intent = state.get("intent", "medical")
        return "medical" if intent == "medical" else "general"
    
    def _route_after_reflection(self, state: QAState) -> str:
        """Route after reflection"""
        reflection_result = state.get("reflection_result", {})
        sufficient = reflection_result.get("sufficient", True)
        retry_count = reflection_result.get("retry_count", 0)
        max_retries = 2
        
        # If sufficient or max retries reached, proceed to answer
        if sufficient or retry_count >= max_retries:
            return "sufficient"
        else:
            return "retry"
    
    async def run(self, query: str, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Execute QA workflow"""
        try:
            self.logger.info(f"Starting QA workflow for query: {query[:50]}...")
            
            # Initialize state
            initial_state = QAState(
                original_query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                intent=None,
                structured_query=None,
                augmented_queries=None,
                context=None,
                reflection_result=None,
                final_context=None,
                answer=None,
                current_step="started",
                is_complete=False,
                error_message=None
            )
            
            # Execute workflow - for now, manually execute the steps since LangGraph sync nodes
            # need to be converted to async properly
            
            # Step 1: Intent classification
            state = await self._intent_classification_node(initial_state)
            
            # Route based on intent
            if state.get("intent") == "medical":
                # Step 2: Query augmentation
                state = await self._query_augmentation_node(state)
                
                # Step 3: Retrieval
                state = await self._retrieval_node(state)
                
                # Step 4: Reflection
                state = await self._reflection_node(state)
                
                # Step 5: Answer generation
                state = await self._answer_generation_node(state)
                
                # Step 6: Save conversation
                state = await self._save_conversation_node(state)
            else:
                # General response
                state = await self._general_response_node(state)
            
            self.logger.info(f"QA workflow completed: {state['current_step']}")
            
            return {
                'success': True,
                'answer': state.get('answer', ''),
                'final_state': state
            }
            
        except Exception as e:
            self.logger.error(f"QA workflow failed: {e}")
            return {
                'success': False,
                'answer': 'Xin lỗi, đã xảy ra lỗi trong quá trình xử lý.',
                'error': str(e)
            }
    
    def get_status(self) -> Dict[str, str]:
        """Get workflow status"""
        return {
            'workflow_type': 'qa',
            'status': 'ready',
            'description': 'Multi-agent Q&A workflow with RAG and reflection'
        }
    
    async def close(self):
        """Close all worker connections"""
        try:
            await self.qa_worker.close()
            await self.save_worker.close()
        except Exception as e:
            self.logger.error(f"Error closing workers: {e}")
