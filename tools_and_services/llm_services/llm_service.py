from .model_loader import ModelLoader
from .prompts import LLMPrompts
import torch
import json
import os
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
from loguru import logger
import asyncio

load_dotenv()

class LLMService:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.model_loader = ModelLoader(self.models_dir)
        self.prompts = LLMPrompts()
        
        # Model configurations from env
        self.model_configs = {
            'query_augmentation': os.getenv('QUERY_AUGMENTATION_MODEL', 'google/flan-t5-base'),
            'summary': os.getenv('SUMMARY_MODEL', 'google/flan-t5-large'),
            'reflection': os.getenv('REFLECTION_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct'),
            'answer': os.getenv('ANSWER_MODEL', 'MedGemma3-4b-it')
        }
    
    async def _generate_text(self, model_dict: Dict, prompt: str, max_length: int = 512) -> str:
        """Generate text using loaded model"""
        try:
            model = model_dict['model']
            tokenizer = model_dict['tokenizer']
            
            # Run tokenization and generation in thread pool
            loop = asyncio.get_event_loop()
            
            inputs = await loop.run_in_executor(
                None,
                lambda: tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            )
            
            with torch.no_grad():
                outputs = await loop.run_in_executor(
                    None,
                    lambda: model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_length,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                )
            
            # Decode only the generated part
            generated_text = await loop.run_in_executor(
                None,
                lambda: tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
            )
            
            return generated_text
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""
    
    async def query_augmentation(self, original_query: str) -> Dict[str, Union[str, List[str]]]:
        """Generate structured query and augmented queries"""
        model_name = self.model_configs['query_augmentation']
        model_dict = await asyncio.get_event_loop().run_in_executor(
            None, self.model_loader.load_model, model_name, True
        )
        
        prompt = self.prompts.query_augmentation_prompt(original_query)
        response = await self._generate_text(model_dict, prompt, max_length=300)
        
        try:
            # Try to parse JSON response
            result = json.loads(response)
            return {
                'structured_query': result.get('structured_query', original_query),
                'augmented_queries': result.get('augmented_queries', [])[:3]  # Max 3 queries
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.warning("Failed to parse JSON from query augmentation, using fallback")
            return {
                'structured_query': original_query,
                'augmented_queries': [original_query]
            }
    
    async def summarize_web_content(self, aug_query: str, web_content: str) -> str:
        """Summarize web content for specific query"""
        model_name = self.model_configs['summary']
        model_dict = await asyncio.get_event_loop().run_in_executor(
            None, self.model_loader.load_model, model_name, True
        )
        
        prompt = self.prompts.summary_web_results_prompt(aug_query, web_content)
        summary = await self._generate_text(model_dict, prompt, max_length=200)
        
        return summary if summary else "Không thể tóm tắt nội dung này."
    
    async def reflection_check(self, structured_query: str, aug_queries: List[str], context: str) -> Dict:
        """Check if context is sufficient to answer queries"""
        model_name = self.model_configs['reflection']
        model_dict = await asyncio.get_event_loop().run_in_executor(
            None, self.model_loader.load_model, model_name, True
        )
        
        prompt = self.prompts.reflection_prompt(structured_query, aug_queries, context)
        response = await self._generate_text(model_dict, prompt, max_length=200)
        
        try:
            result = json.loads(response)
            return {
                'sufficient': result.get('sufficient', False),
                'reasoning': result.get('reasoning', ''),
                'follow_up_queries': result.get('follow_up_queries', [])[:2]  # Max 2 follow-up
            }
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from reflection, assuming insufficient")
            return {
                'sufficient': False,
                'reasoning': 'Không thể đánh giá đầy đủ thông tin',
                'follow_up_queries': []
            }
    
    async def generate_answer(self, original_query: str, context: str, chat_history: List[Dict] = None) -> str:
        """Generate final answer using context and chat history"""
        model_name = self.model_configs['answer']
        model_dict = await asyncio.get_event_loop().run_in_executor(
            None, self.model_loader.load_model, model_name, True
        )
        
        chat_history = chat_history or []
        prompt = self.prompts.answer_prompt(original_query, context, chat_history)
        
        answer = await self._generate_text(model_dict, prompt, max_length=500)
        return answer if answer else "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này."
    
    def generate_general_response(self, query: str) -> str:
        """Generate response for non-medical queries"""
        return self.prompts.general_prompt(query)
    
    async def preload_models(self, model_types: List[str] = None):
        """Preload specified models to memory"""
        if model_types is None:
            model_types = list(self.model_configs.keys())
        
        tasks = []
        for model_type in model_types:
            if model_type in self.model_configs:
                model_name = self.model_configs[model_type]
                logger.info(f"Preloading {model_type} model: {model_name}")
                task = asyncio.get_event_loop().run_in_executor(
                    None, self.model_loader.load_model, model_name, True
                )
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get loading status of all configured models"""
        loaded_models = self.model_loader.get_loaded_models()
        status = {}
        
        for model_type, model_name in self.model_configs.items():
            status[model_type] = model_name in loaded_models
        
        return status
