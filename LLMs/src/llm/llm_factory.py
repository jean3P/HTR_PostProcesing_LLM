# src/llm/llm_factory.py

from src.llm.mistral_llm import MistralLLM


class LLMFactory:
    @staticmethod
    def get_llm(model_name: str):
        """
        Return a Mistral LLM instance based on the model name.
        """
        if model_name.startswith("mistral"):
            return MistralLLM(model_name)
        else:
            raise ValueError(f"Unknown LLM model name: {model_name}")
