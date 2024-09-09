# src/prompts/mistral_prompts.py

def generate_mistral_prompt(prompt_name: str, **kwargs):
    if prompt_name == "correct_text":
        ocr_text = kwargs.get('ocr_text')
        suggestions = kwargs.get('suggestions')
        return f"Correct this OCR text: {ocr_text} using these suggestions: {suggestions}"

    elif prompt_name == "evaluate_text":
        original_text = kwargs.get('original_text')
        corrected_text = kwargs.get('corrected_text')
        return f"Evaluate the correctness of this text: {corrected_text} compared to: {original_text}"

    else:
        raise ValueError(f"Unknown prompt name: {prompt_name}")
