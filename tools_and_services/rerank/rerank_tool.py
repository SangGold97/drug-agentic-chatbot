import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
from loguru import logger
import asyncio
from time import time

load_dotenv()

class RerankTool:
    def __init__(self):
        self.model_name = os.getenv('RERANK_MODEL', 'Qwen/Qwen3-Reranker-0.6B')
        self.model = None
        self.tokenizer = None
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        self.max_length = 1024
        
        # Task instruction for drug/disease/gene reranking
        self.task_instruction = "Given a medical query about drugs, diseases, or genes, determine if the document is relevant to answer the query."
        
        # Prefix and suffix for Qwen3-Reranker format
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    def load_model(self):
        """Load Qwen3-Reranker model"""
        if self.model is None:
            try:
                logger.info(f"Loading rerank model: {self.model_name}")
                start = time()

                # Create cache directory if it doesn't exist
                os.makedirs(self.cache_dir, exist_ok=True)
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    padding_side='left',
                    cache_dir=self.cache_dir
                )
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=self.cache_dir
                ).eval()
                
                # Get token IDs for yes/no
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
                
                # Encode prefix and suffix tokens
                self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
                self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
                
                load_time = time() - start
                logger.info(f"Rerank model loaded successfully in {load_time:.2f}s")

            except Exception as e:
                logger.error(f"Failed to load rerank model: {e}")
                raise
    
    def format_instruction(self, query: str, document: str) -> str:
        """Format instruction for Qwen3-Reranker with medical context"""
        return f"<Instruct>: {self.task_instruction}\n<Query>: {query}\n<Document>: {document}"
    
    def process_inputs(self, pairs: List[str]) -> Dict:
        """Process input pairs for Qwen3-Reranker"""
        
        # Run tokenization
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            return_attention_mask=False
        )
        
        # Add prefix and suffix tokens
        for i, input_ids in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + input_ids + self.suffix_tokens
        
        # Pad to max_length
        inputs = self.tokenizer.pad(
            inputs, 
            padding=True, 
            # max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs
    
    @torch.no_grad()
    async def compute_logits(self, inputs: Dict) -> List[float]:
        """Compute relevance scores using Qwen3-Reranker (async for GPU operations)"""
        loop = asyncio.get_event_loop()
        
        # Run model inference in thread pool to avoid blocking
        def _inference():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores
        
        scores = await loop.run_in_executor(None, _inference)
        return scores

    def elbow_pruning(self, chunks: List[Dict]) -> List[Dict]:
        """
        Prune chunks using elbow method based on score intervals.
        Returns chunks from start to the position with the largest interval.
        """
        if not chunks or len(chunks) <= 2:
            return chunks
        
        try:
            # Extract scores from chunks
            scores = []
            for chunk in chunks:
                # Try to get score from different possible field names
                score = chunk.get('rerank_score', 0)
                scores.append(float(score))
            
            # Calculate intervals between consecutive scores
            intervals = []
            for i in range(len(scores) - 1):
                interval = abs(scores[i] - scores[i + 1])
                intervals.append(interval)
            
            # Find the position with the largest interval
            max_interval_idx = intervals.index(max(intervals))
            
            # Return chunks from start to the position with largest interval (inclusive)
            pruned_chunks = chunks[:max_interval_idx + 1]
            logger.info(f"Elbow pruned: {len(chunks)} -> {len(pruned_chunks)} chunks")
            
            return pruned_chunks
            
        except Exception as e:
            logger.error(f"Failed to perform elbow pruning: {e}")
            return chunks
        
    async def rerank(self, query: str, chunks: List[Dict], 
                     top_k: int = 5, elbow: bool = True) -> List[Dict]:
        """Rerank chunks based on query relevance using Qwen3-Reranker"""
        if not chunks:
            return chunks
        
        if self.model is None:
            self.load_model()
        
        try:
            # Prepare query-chunk pairs for reranking
            pairs = []
            for chunk in chunks:
                content = chunk.get('content', '')
                formatted_pair = self.format_instruction(query, content)
                pairs.append(formatted_pair)
            
            # Get relevance scores using async pipeline
            inputs = self.process_inputs(pairs)
            scores = await self.compute_logits(inputs)
            
            # Combine chunks with scores and sort
            chunks_with_scores = []
            for i, chunk in enumerate(chunks):
                chunk_copy = chunk.copy()
                chunk_copy['rerank_score'] = float(scores[i])
                chunks_with_scores.append(chunk_copy)
            
            # Sort by rerank score (descending)
            reranked_chunks = sorted(
                chunks_with_scores, 
                key=lambda x: x['rerank_score'], 
                reverse=True
            )

            # Hard limit if top_k is specified
            if top_k is not None:
                reranked_chunks = reranked_chunks[:top_k]

            # Elbow pruning if enabled
            if elbow:
                reranked_chunks = self.elbow_pruning(reranked_chunks)
            
            logger.info(f"Reranked completed with {len(reranked_chunks)} chunks from {len(chunks)} total")
            return reranked_chunks
        
        except Exception as e:
            logger.error(f"Failed to rerank chunks: {e}")
            return chunks

    def health_check(self) -> Dict[str, str]:
        """Health check for rerank service"""
        try:
            if self.model is None:
                return {"status": "loading", "message": "Model not loaded"}
            return {"status": "healthy", "message": "Rerank service is ready"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}


# Test the RerankTool
if __name__ == "__main__":
    async def test_rerank_tool():
        start = time()
        rerank_tool = RerankTool()
        rerank_tool.load_model()
        print(f"Rerank model loaded successfully in {(time() - start):.2f}s")
        start = time()

        # Example query and chunks for medical domain
        query = "Relation of liver glucose production to type 2 diabetes management"
        chunks = [
            {"content": "Metformin is a first-line medication for type 2 diabetes management and works by reducing glucose production in the liver."},
            {"content": "Diet and regular exercise are fundamental lifestyle interventions for managing type 2 diabetes effectively."},
            {"content": "Insulin therapy may be required for patients with type 2 diabetes who cannot achieve glycemic control with oral medications."},
            {"content": "GLP-1 receptor agonists like liraglutide have shown efficacy in improving glycemic control and promoting weight loss in diabetic patients."},
            {"content": "Regular monitoring of HbA1c levels is essential for assessing long-term diabetes management and treatment effectiveness."},
            {"content": "SGLT2 inhibitors such as empagliflozin provide cardiovascular benefits in addition to glucose-lowering effects for diabetic patients."},
            {"content": "Aspirin is commonly used for cardiovascular disease prevention but is not a primary diabetes treatment."}
        ]
        
        print("Testing Qwen3-Reranker with medical content...")
        reranked_chunks = await rerank_tool.rerank(query, chunks, top_k=8, elbow=True)
        
        print(f"\nQuery: {query}")
        print("\nReranked Results:")
        for i, chunk in enumerate(reranked_chunks):
            print(f"{i+1}. Score: {chunk['rerank_score']:.4f}")
            print(f"   Content: {chunk['content']}")
            print()
        inference_time = time() - start
        print(f"Total time inference: {inference_time:.2f} s")

    asyncio.run(test_rerank_tool())