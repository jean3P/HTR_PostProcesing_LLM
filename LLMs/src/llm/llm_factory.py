# src/llm/llm_factory.py

from llm.gpt_llm import GptLLM
from llm.mistral_llm import MistralLLM
from llm.openrouter_llm import OpenRouterLLM


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
        # OpenRouter models
        elif model_name in [
            "qwen-2.5-vl-72b",
            "phi-4",
            "internalvl3",
            "mistral",
            "gemini-2.5-pro",
            "gpt-4.1-mini",
            "claude-sonnet-4"
        ]:
            return OpenRouterLLM(model_name)

        else:
            raise ValueError(f"Unknown LLM model name: {model_name}")
