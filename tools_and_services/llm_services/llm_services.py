try:
    from .prompts import LLMPrompts
except ImportError:
    from prompts import LLMPrompts
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import json
import os
from typing import Dict, Generator
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

class LLMService:
    def __init__(self, service_name: str):
        """Initialize LLMService with specific service name
        
        Args:
            service_name: One of 'structured_query_generator', 'reflection', 'general', 'answer'
        """
        if service_name not in ['structured_query_generator', 'reflection', 'general', 'answer']:
            raise ValueError(f"Invalid service name: {service_name}. Must be one of: structured_query_generator, reflection, general, answer")

        self.service_name = service_name
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.model_name = os.getenv('STRUCTURED_QUERY_GENERATOR_MODEL') if service_name == 'structured_query_generator' else \
                          os.getenv('REFLECTION_MODEL') if service_name == 'reflection' else \
                          os.getenv('GENERAL_MODEL') if service_name == 'general' else \
                          os.getenv('ANSWER_MODEL', 'google/medgemma-4b-it')
        self.prompts = LLMPrompts()
        
        # Load model and processor
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load MedGemma model from cache directory"""
        try:
            local_dir = os.path.join(self.cache_dir, self.model_name.replace('/', '_'))
            
            if not os.path.exists(local_dir):
                raise FileNotFoundError(f"Model not found in cache: {local_dir}")
            
            # Load processor and model for MedGemma
            self.processor = AutoProcessor.from_pretrained(local_dir)
            self.model = AutoModelForImageTextToText.from_pretrained(
                local_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa"
            )
            
            logger.info(f"MedGemma model loaded successfully for service: {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to load MedGemma model for {self.service_name}: {e}")
            raise
    
    def _get_prompt(self, *args) -> str:
        """Get appropriate prompt based on service name"""
        if self.service_name == 'structured_query_generator':
            return self.prompts.structured_query_prompt(args[0])
        elif self.service_name == 'reflection':
            return self.prompts.reflection_prompt(args[0], args[1])
        elif self.service_name == 'general':
            return self.prompts.general_prompt(args[0])
        elif self.service_name == 'answer':
            return self.prompts.answer_prompt(args[0], args[1], args[2])
        else:
            raise ValueError(f"Unknown service: {self.service_name}")
    
    def generate_response(self, *args) -> str:
        """Generate response using MedGemma model"""
        try:
            prompt = self._get_prompt(*args)
            
            messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": "Bạn là một chuyên gia y tế, dược và di truyền, bạn có khả năng tóm tắt thông tin y học và trả lời câu hỏi theo hướng dẫn."}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=1024,
                    do_sample=True, 
                    temperature=0.7, 
                    top_p=0.9
                )
                generation = generation[0][input_len:]
            
            generated_text = self.processor.decode(generation, skip_special_tokens=True)
            
            # Clean up JSON response for structured_query_generator and reflection services
            if self.service_name in ['structured_query_generator', 'reflection']:
                # Remove markdown code blocks if present
                if '```json' in generated_text:
                    generated_text = generated_text.split('```json')[1].split('```')[0].strip()
                elif '```' in generated_text:
                    generated_text = generated_text.split('```')[1].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to generate response for {self.service_name}: {e}")
            return ""
    
