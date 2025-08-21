# src/llm/mistral_llm.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from dotenv import load_dotenv
from llm.base_llm import BaseLLM

load_dotenv()

TOKEN_HUGGING_FACE = os.getenv('TOKEN')


class MistralLLM(BaseLLM):
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        super().__init__(model_name)

        # Check device
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Only use quantization and device_map on GPU
        if device == 'cuda':
            try:
                from transformers import BitsAndBytesConfig
                self.quantization = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                self.quantization = None
        else:
            self.quantization = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=TOKEN_HUGGING_FACE
        )

        # Initialize model - NO device_map for CPU
        if device == 'cuda' and self.quantization:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=TOKEN_HUGGING_FACE,
                quantization_config=self.quantization,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            # CPU mode - no device_map, no quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=TOKEN_HUGGING_FACE,
                torch_dtype=torch.float32,
            )
            if device == 'cpu':
                self.model = self.model.to('cpu')

