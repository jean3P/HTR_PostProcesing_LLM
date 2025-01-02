# src/llm/mistral_llm.py

import os
import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from dotenv import load_dotenv
from llm.base_llm import BaseLLM

load_dotenv()

TOKEN_HUGGING_FACE = os.getenv('TOKEN')


class GemmaLLM(BaseLLM):
    def __init__(self, model_name="google/gemma-7b"):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=TOKEN_HUGGING_FACE,
            torch_dtype=torch.float16,  # Make sure the dtype is consistent
            device_map="auto",
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.float16},  # Consistent with the model
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]