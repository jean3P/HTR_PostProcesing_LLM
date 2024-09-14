# src/llm/base_llm.py

class BaseLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.stream = bool

