# repos/HTR_PostProcesing_LLM/LLMs/src/prompts/openrouter/openrouter_response_parser.py

import re
from typing import Tuple, Optional
import logging


class OpenRouterResponseParser:
    """Parser for cleaning and extracting corrected text from OpenRouter LLM responses."""

    @staticmethod
    def extract_corrected_text(raw_response: str, original_text: str, logger: logging.Logger) -> str:
        """
        Extract the corrected text from various response formats.

        Args:
            raw_response: The raw response from the LLM
            original_text: The original OCR text for fallback
            logger: Logger instance

        Returns:
            The cleaned corrected text
        """
        if not raw_response:
            return original_text

        # Method 1: Look for quoted text after key phrases
        patterns = [
            # Pattern 1: "The corrected version of the text line is: "..."
            r'[Tt]he corrected (?:version of the )?(?:text line|version) is:\s*["\']([^"\']+)["\']',
            # Pattern 2: "The corrected text line is: ..."
            r'[Tt]he corrected text line is:\s*["\']([^"\']+)["\']',
            # Pattern 3: "corrected version: "..."
            r'corrected version:\s*["\']([^"\']+)["\']',
            # Pattern 4: 'The corrected version would be: "..."'
            r'[Tt]he corrected version would be:\s*["\']([^"\']+)["\']',
            # Pattern 5: Direct quotes at the beginning
            r'^["\']([^"\']+)["\']',
            # Pattern 6: After "is:" without quotes
            r'[Tt]he corrected (?:text line|version) is:\s*([^."\n]+)',
            # Pattern 7: "Then the corrected text line is: ..."
            r'[Tt]hen the corrected text line is:\s*["\']?([^"\'.\n]+)["\']?',
        ]

        for pattern in patterns:
            match = re.search(pattern, raw_response, re.IGNORECASE | re.DOTALL)
            if match:
                corrected_text = match.group(1).strip()
                logger.debug(f"Matched pattern: {pattern}")
                logger.debug(f"Extracted text: {corrected_text}")

                # Clean up any remaining artifacts
                corrected_text = OpenRouterResponseParser._clean_extracted_text(corrected_text)

                # Validate the extraction
                if corrected_text and len(corrected_text) > 0:
                    return corrected_text

        # Method 2: If no pattern matched, try to extract text between quotes
        quote_pattern = r'"([^"]+)"'
        quotes = re.findall(quote_pattern, raw_response)
        if quotes:
            # Filter out meta-text and find the most likely corrected version
            for quote in quotes:
                # Skip if it's clearly meta-text
                if any(phrase in quote.lower() for phrase in [
                    'the corrected', 'here\'s', 'explanation', 'correction',
                    'should be', 'appears to', 'likely', 'seems'
                ]):
                    continue

                # Check if it's similar in length to original (within reasonable bounds)
                if 0.5 <= len(quote) / len(original_text) <= 2.0:
                    cleaned = OpenRouterResponseParser._clean_extracted_text(quote)
                    if cleaned:
                        logger.debug(f"Extracted from quotes: {cleaned}")
                        return cleaned

        # Method 3: Try to find the corrected text in a structured response
        # Sometimes the response has "Original: ... Corrected: ..."
        corrected_match = re.search(r'[Cc]orrected:\s*([^.\n]+)', raw_response)
        if corrected_match:
            corrected_text = corrected_match.group(1).strip()
            corrected_text = OpenRouterResponseParser._clean_extracted_text(corrected_text)
            if corrected_text:
                logger.debug(f"Extracted from 'Corrected:' format: {corrected_text}")
                return corrected_text

        # Method 4: If response is very short and doesn't match patterns, it might be the answer
        lines = raw_response.strip().split('\n')
        if len(lines) == 1 and len(raw_response) < 200 and not any(
                phrase in raw_response.lower() for phrase in [
                    'the corrected', 'appears to', 'likely', 'should be',
                    'here\'s', 'explanation', 'breakdown'
                ]
        ):
            cleaned = OpenRouterResponseParser._clean_extracted_text(raw_response)
            if cleaned and 0.5 <= len(cleaned) / len(original_text) <= 2.0:
                logger.debug(f"Using full response as corrected text: {cleaned}")
                return cleaned

        # If all else fails, log warning and return original
        logger.warning(f"Could not extract corrected text from response. Using original text.")
        logger.debug(f"Failed response: {raw_response[:200]}...")
        return original_text

    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """
        Clean up the extracted text by removing common artifacts.

        Args:
            text: The extracted text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return text

        # Remove any leading/trailing quotes
        text = text.strip('"\'')

        # Remove common prefixes that might have been included
        prefixes_to_remove = [
            "The corrected text line is:",
            "The corrected version is:",
            "Corrected text line:",
            "Corrected:",
            "Then the corrected text line is:",
            "The text line is:",
        ]

        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Remove any trailing explanation markers
        if "Here's" in text or "Explanation:" in text:
            text = text.split("Here's")[0].split("Explanation:")[0].strip()

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove any trailing punctuation that seems like it's part of the explanation
        text = re.sub(r'[.,:;]+\s*$', '', text)

        # If the text ends with incomplete explanation, remove it
        if text.endswith(" to") or text.endswith(" is") or text.endswith(" was"):
            text = ' '.join(text.split()[:-1])

        return text

    @staticmethod
    def extract_confidence_and_justification(
            raw_response: str,
            logger: logging.Logger
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract confidence score and justification from evaluation response.

        Args:
            raw_response: The raw response from the LLM
            logger: Logger instance

        Returns:
            Tuple of (confidence, justification) or (None, None) if not found
        """
        try:
            # Look for confidence score
            confidence_patterns = [
                r'[Cc]onfidence:\s*(\d+)',
                r'[Cc]onfidence\s*[Ss]core:\s*(\d+)',
                r'[Cc]onfidence:\s*(\d+)%',
                r'[Cc]onfidence\s*=\s*(\d+)',
            ]

            confidence = None
            for pattern in confidence_patterns:
                match = re.search(pattern, raw_response)
                if match:
                    confidence = match.group(1)
                    break

            # Look for justification
            justification_patterns = [
                r'[Jj]ustification:\s*([^\n]+(?:\n(?![A-Z][a-z]*:)[^\n]+)*)',
                r'[Rr]easoning:\s*([^\n]+(?:\n(?![A-Z][a-z]*:)[^\n]+)*)',
                r'[Ee]xplanation:\s*([^\n]+(?:\n(?![A-Z][a-z]*:)[^\n]+)*)',
            ]

            justification = None
            for pattern in justification_patterns:
                match = re.search(pattern, raw_response, re.MULTILINE)
                if match:
                    justification = match.group(1).strip()
                    # Clean up the justification
                    justification = re.sub(r'\s+', ' ', justification)
                    # Limit length
                    if len(justification) > 500:
                        justification = justification[:497] + "..."
                    break

            # If we found confidence but no justification, try to extract any explanation
            if confidence and not justification:
                # Remove the confidence line and see what's left
                remaining = re.sub(r'[Cc]onfidence:?\s*\d+%?\s*', '', raw_response).strip()
                if remaining and len(remaining) > 20:
                    justification = remaining[:500] if len(remaining) > 500 else remaining
                    justification = re.sub(r'\s+', ' ', justification)

            if confidence and justification:
                logger.debug(f"Extracted confidence: {confidence}, justification length: {len(justification)}")
                return confidence, justification
            else:
                logger.debug(f"Could not extract confidence/justification from response")
                return None, None

        except Exception as e:
            logger.error(f"Error extracting confidence/justification: {e}")
            return None, None

    @staticmethod
    def clean_text_line(original_text: str, corrected_text: str) -> str:
        """
        Additional cleaning to ensure the corrected text is properly formatted.
        Matches the functionality of clean_text from utils.aux_processing

        Args:
            original_text: The original OCR text
            corrected_text: The corrected text

        Returns:
            Cleaned corrected text
        """
        # If corrected text is empty or None, return original
        if not corrected_text:
            return original_text

        # Remove multiple spaces
        corrected_text = re.sub(r'\s+', ' ', corrected_text).strip()

        # Ensure we're not returning meta-text
        meta_phrases = [
            "the corrected", "appears to", "likely", "should be",
            "here's", "explanation", "breakdown", "correction",
            "assuming", "context", "plausible", "reasonable"
        ]

        # Check if the corrected text is mostly meta-text
        lower_text = corrected_text.lower()
        if any(phrase in lower_text for phrase in meta_phrases) and len(corrected_text) > 100:
            # This might be an explanation rather than the corrected text
            # Try one more extraction
            match = re.search(r'"([^"]+)"', corrected_text)
            if match:
                corrected_text = match.group(1)

        # Final validation - ensure it's not drastically different in length
        if len(corrected_text) > 3 * len(original_text) or len(corrected_text) < len(original_text) / 3:
            # Unless it's a very short text, this might be wrong
            if len(original_text) > 10:
                return original_text

        return corrected_text.strip()
