# src/llm/llm_factory.py
from llm.gpt_llm import GptLLM
from llm.mistral_llm import MistralLLM


class LLMFactory:
    @staticmethod
    def get_llm(model_name: str):
        """
        Return a Mistral LLM instance based on the model name.
        """
        if model_name.startswith("mistral"):
            return MistralLLM(model_name)
        elif model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
            return GptLLM(model_name)
        else:
            raise ValueError(f"Unknown LLM model name: {model_name}")
