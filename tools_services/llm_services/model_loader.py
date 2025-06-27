from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from typing import Dict, Optional
import logging

class ModelLoader:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.logger = logging.getLogger(__name__)
        
        # 4-bit quantization config
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Model cache
        self._loaded_models: Dict[str, Dict] = {}
    
    def download_model(self, model_name: str) -> str:
        """Download model from Hugging Face"""
        local_dir = os.path.join(self.models_dir, model_name.replace('/', '_'))
        
        if os.path.exists(local_dir):
            self.logger.info(f"Model {model_name} already exists locally")
            return local_dir
        
        try:
            self.logger.info(f"Downloading model {model_name}")
            snapshot_download(
                repo_id=model_name,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            self.logger.info(f"Model {model_name} downloaded successfully")
            return local_dir
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}")
            raise
    
    def load_model(self, model_name: str, use_quantization: bool = True) -> Dict:
        """Load model and tokenizer"""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
        
        try:
            local_dir = self.download_model(model_name)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with or without quantization
            if use_quantization:
                model = AutoModelForCausalLM.from_pretrained(
                    local_dir,
                    quantization_config=self.quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    local_dir,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            
            # Cache the loaded model
            model_dict = {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_name
            }
            self._loaded_models[model_name] = model_dict
            
            self.logger.info(f"Model {model_name} loaded successfully")
            return model_dict
        
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def unload_model(self, model_name: str):
        """Unload model from memory"""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            torch.cuda.empty_cache()
            self.logger.info(f"Model {model_name} unloaded")
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded models"""
        return list(self._loaded_models.keys())
