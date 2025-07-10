try:
    from .prompts import LLMPrompts
except ImportError:
    from prompts import LLMPrompts
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer
import torch
import json
import os
import asyncio
from typing import Dict, Generator, AsyncGenerator
from loguru import logger
from dotenv import load_dotenv
from threading import Thread
load_dotenv()

class LLMService:
    def __init__(self):
        """Initialize LLMService"""

        self.cache_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.model_name = os.getenv('ANSWER_MODEL', 'google/medgemma-4b-it')
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

            logger.info("MedGemma model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MedGemma model: {e}")
            raise
    
    def _get_prompt(self, service_name: str, *args) -> str:
        """Get appropriate prompt based on service name"""
        if service_name == 'structured_query_generator':
            return self.prompts.structured_query_prompt(args[0])
        elif service_name == 'reflection':
            return self.prompts.reflection_prompt(args[0], args[1])
        elif service_name == 'general':
            return self.prompts.general_prompt(args[0], args[1])
        elif service_name == 'answer':
            return self.prompts.answer_prompt(args[0], args[1], args[2])
        else:
            raise ValueError(f"Unknown service: {service_name}")

    async def generate_response(self, service_name: str, *args) -> str:
        """Generate response using MedGemma model"""
        try:
            prompt = self._get_prompt(service_name, *args)
            logger.info(f"Prompt:\n{prompt}")

            messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": self.prompts.system_prompt()}
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
            
            # Run model generation in thread pool to avoid blocking
            def _generate():
                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs, 
                        max_new_tokens=1024,
                        do_sample=True, 
                        temperature=0.7, 
                        top_p=0.9
                    )
                    return generation[0][input_len:]
            
            loop = asyncio.get_event_loop()
            generation = await loop.run_in_executor(None, _generate)
            
            generated_text = self.processor.decode(generation, skip_special_tokens=True)
            
            # Clean up JSON response for structured_query_generator and reflection services
            if service_name in ['structured_query_generator', 'reflection']:
                # Remove markdown code blocks if present
                if '```json' in generated_text:
                    generated_text = generated_text.split('```json')[1].split('```')[0].strip()
                elif '```' in generated_text:
                    generated_text = generated_text.split('```')[1].strip()
            
            # Clean up VRAM
            del inputs, generation
            torch.cuda.empty_cache()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to generate response for {service_name}: {e}")
            return ""

    async def generate_stream_response(self, service_name: str, *args) -> AsyncGenerator[str, None]:
        """Generate streaming response using MedGemma model with TextIteratorStreamer"""
        try:
            prompt = self._get_prompt(service_name, *args)
            logger.info(f"Streaming prompt for {service_name}")

            messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": self.prompts.system_prompt()}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            
            # Create TextIteratorStreamer for streaming output
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=None
            )
            
            # Generation parameters with streamer
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "streamer": streamer
            }
            
            # Run generation in separate thread to avoid blocking
            def _generate():
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            
            thread = Thread(target=_generate)
            thread.start()
            
            # Stream tokens as they are generated
            for new_text in streamer:
                yield new_text
                
            thread.join()
            
            # Clean up VRAM
            del inputs
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to stream response for {service_name}: {e}")
            yield ""
    
    def health_check(self) -> Dict[str, str]:
        """Health check for LLM services"""
        try:
            return {"status": "healthy", "message": "LLM services are ready"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}