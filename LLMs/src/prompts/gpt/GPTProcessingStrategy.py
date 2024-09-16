class TextProcessingStrategy:
    def get_name_method(self):
        """
        Return the name of the method being used. This should be implemented by each specific strategy.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def check_and_correct_text_line(self, text_line, *args, **kwargs):
        """
        Check and correct a given text line.
        Each specific strategy (Mistral, GPT) must implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def correct_with_suggestions(self, ocr_text, suggestions, *args, **kwargs):
        """
        Correct the OCR text line based on given suggestions.
        Each specific strategy should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def correct_duplicated_words(self, text_line, *args, **kwargs):
        """
        Correct duplicated words in the text line.
        Each specific strategy should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate_corrected_text(self, original_text, corrected_text, *args, **kwargs):
        """
        Evaluate the corrected text, providing confidence and justification for the corrections.
        This should be implemented by each specific strategy.
        """
        raise NotImplementedError("Subclasses should implement this method.")
