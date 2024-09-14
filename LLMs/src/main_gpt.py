from llm.llm_factory import LLMFactory
from utils.io_utils import get_latest_result_for_datasets

llm_name_2 = "gpt-3.5-turbo-instruct"
gpt_llm = LLMFactory.get_llm(llm_name_2)
pipe = gpt_llm.pipe
mistral_tokenizer = gpt_llm.tokenizer
llms = ['gpt-3.5']
model_ocr = "Flor_model"
datasets = ['washington']
train_sizes = ['train_25']
train_suggestion = ['']
latest_results = get_latest_result_for_datasets(llms, datasets, train_sizes, model_ocr)