from huggingface_hub import snapshot_download
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          BitsAndBytesConfig, 
                          T5Tokenizer, 
                          T5ForConditionalGeneration,
                          AutoProcessor, 
                          AutoModelForImageTextToText,
                          Gemma3ForCausalLM
                          )
import torch
import os
from typing import Dict, Optional
from loguru import logger
from prompts import LLMPrompts
from time import time

llmprompts = LLMPrompts()

class ModelLoader:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 8-bit quantization config
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        # Model cache
        self._loaded_models: Dict[str, Dict] = {}
    
    def download_model(self, model_name: str) -> str:
        """Download model from Hugging Face"""
        local_dir = os.path.join(self.models_dir, model_name.replace('/', '_'))
        
        if os.path.exists(local_dir):
            logger.info(f"Model {model_name} already exists locally")
            return local_dir
        
        try:
            logger.info(f"Downloading model {model_name}")
            snapshot_download(
                repo_id=model_name,
                local_dir=local_dir,
                # local_dir_use_symlinks=False
            )
            logger.info(f"Model {model_name} downloaded successfully")
            return local_dir
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            raise
    
    def load_medgemma_model(self, model_name: str) -> Dict:
        """Load MedGemma model and processor"""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
        
        try:
            local_dir = self.download_model(model_name)
            
            # Load processor and model for MedGemma
            processor = AutoProcessor.from_pretrained(local_dir)
            model = AutoModelForImageTextToText.from_pretrained(
                local_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa"  # Better for Turing GPU (RTX 5000)
            )
            
            # Cache the loaded model
            model_dict = {
                'model': model,
                'processor': processor,
                'model_name': model_name
            }
            self._loaded_models[model_name] = model_dict
            logger.info(f"MedGemma model {model_name} loaded successfully")

            # Test generate capability
            start = time()
            query = "Bị đau đầu, tôi có nên tiếp tục làm việc căng thẳng không hay nên nghỉ ngơi?"
            logger.info(f"Testing generation with query: {query}")
            prompt = llmprompts.general_prompt(query)
            messages = [
                {"role": "system", "content": [
                    {"type": "text", "text": "Bạn là một chuyên gia y tế."}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": f"{prompt}"}
                ]}
            ]
            
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=1024, 
                                            do_sample=True, temperature=0.6, top_p=0.9)
                generation = generation[0][input_len:]
            
            generated_text = processor.decode(generation, skip_special_tokens=True)
            logger.info(f"Test generation successful: {generated_text}")
            inference_time = time() - start
            logger.info(f"Inference time: {inference_time:.2f} seconds")
            return model_dict
        
        except Exception as e:
            logger.error(f"Failed to load MedGemma model {model_name}: {e}")
            raise

    def load_qwen3_model(self, model_name: str) -> Dict:
        """Load Qwen3 model"""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
        
        try:
            local_dir = self.download_model(model_name)

            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            model = AutoModelForCausalLM.from_pretrained(
                local_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa"  # Optimized for better performance
            )
            
            # Cache the loaded model
            model_dict = {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_name
            }
            self._loaded_models[model_name] = model_dict
            logger.info(f"Qwen3 model {model_name} loaded successfully")

            # Test generate capability
            start = time()
            prompt = "Tôi đang bị đau đầu, tôi có nên uống paracetamol không?"
            logger.info(f"Testing generation with prompt: {prompt}")

            prompt = llmprompts.query_augmentation_prompt(prompt)
            messages = [
                {"role": "system", "content": "Bạn là một chuyên gia y tế."},
                {"role": "user", "content": prompt}
            ]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, 
                tokenize=True,
                enable_thinking=False,
                return_dict=True, return_tensors="pt"
            ).to(model.device)
            
            outputs = model.generate(**inputs, max_new_tokens=256,
                                     do_sample=True, temperature=0.7, top_p=0.8)
            outputs = outputs[0][len(inputs['input_ids'][0]):]  # Skip input part
            outputs = tokenizer.decode(outputs, skip_special_tokens=True)
            inference_time = time() - start
            logger.info(f"Test generation successful: {outputs}")
            logger.info(f"Inference time: {inference_time:.2f} seconds")

            return model_dict
        
        except Exception as e:
            logger.error(f"Failed to load Qwen3 model {model_name}: {e}")
            raise

    def load_gemma3_model(self, model_name: str) -> Dict:
        """Load Gemma3 model"""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
        
        try:
            local_dir = self.download_model(model_name)
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            model = Gemma3ForCausalLM.from_pretrained(
                local_dir,
                # quantization_config=self.quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa"  # Optimized for better performance
            ).eval()
            
            # Cache the loaded model
            model_dict = {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_name
            }
            self._loaded_models[model_name] = model_dict
            logger.info(f"Gemma3 model {model_name} loaded successfully")

            # Test generate capability
            start = time()
            prompt = "Tôi đang bị đau đầu, tôi nên dùng thuốc gì?"
            logger.info(f"Testing generation with prompt: {prompt}")
            prompt = llmprompts.query_augmentation_prompt(prompt)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=256)
            
            # Only decode the newly generated tokens, excluding the input
            generated_tokens = outputs[0][input_len:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            inference_time = time() - start
            logger.info(f"Test generation successful: {generated_text}")
            logger.info(f"Inference time: {inference_time:.2f} seconds")
            return model_dict
        
        except Exception as e:
            logger.error(f"Failed to load Gemma3 model {model_name}: {e}")
            raise

    def load_model(self, model_name: str, use_quantization: bool = True) -> Dict:
        """Load model and tokenizer"""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        if "medgemma" in model_name.lower():
            return self.load_medgemma_model(model_name)
        elif "qwen" in model_name.lower():
            return self.load_qwen3_model(model_name)
        elif "gemma-3" in model_name.lower():
            return self.load_gemma3_model(model_name)
        else:
            logger.error(f"Unsupported model type: {model_name}")
            raise ValueError(f"Unsupported model type: {model_name}")
    
    def unload_model(self, model_name: str):
        """Unload model from memory"""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            torch.cuda.empty_cache()
            logger.info(f"Model {model_name} unloaded")

    def get_loaded_models(self) -> list:
        """Get list of currently loaded models"""
        return list(self._loaded_models.keys())
    

if __name__ == "__main__":
    loader = ModelLoader()

    model_answer = loader.load_model("google/medgemma-4b-it")
    print(f"Loaded model: {model_answer['model_name']}")

    # model_reflection = loader.load_model("Qwen/Qwen3-4B-AWQ")
    # print(f"Loaded model: {model_reflection['model_name']}")

    # model_gemma3 = loader.load_model("google/gemma-3-1b-it")
    # print(f"Loaded model: {model_gemma3['model_name']}")
