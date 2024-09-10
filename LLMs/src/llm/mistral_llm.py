# src/llm/mistral_llm.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from src.llm.base_llm import BaseLLM

from dotenv import load_dotenv

load_dotenv()

TOKEN_HUGGING_FACE = os.getenv('TOKEN')


class MistralLLM(BaseLLM):
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        super().__init__(model_name)
        self.quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=TOKEN_HUGGING_FACE
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=TOKEN_HUGGING_FACE,
            quantization_config=self.quantization,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=10,
            truncation=True,  # Ensure truncation is enabled
            max_length=32  # Use the max_length setting here for truncation
        )
