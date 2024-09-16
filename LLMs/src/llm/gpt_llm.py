import os

from dotenv import load_dotenv

from llm.base_llm import BaseLLM

load_dotenv()

#OPENAI_API_KEY = os.getenv("TOKEN_OPENAI_ANNA")
OPENAI_API_KEY = os.getenv("TOKEN_OPEN_AI_PERSONAL")

class GptLLM(BaseLLM):
    def __init__(self, model_name="gpt-3.5-turbo-instruct"):
        super().__init__(model_name)
        self.stream = True
        self.openai_token = OPENAI_API_KEY
