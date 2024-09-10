import re


def correct_punctuation_spacing(text):
    """
    Correct the spacing of punctuation marks in the text.
    Ensures proper spacing around punctuation marks, parentheses, quotes, and brackets.
    This includes characters like . , ; : ! ? as well as (), "", '', {}, [], and |.
    """

    # Remove spaces before punctuation marks like . , ; : ! ?
    text = re.sub(r'\s([?.!,;:])', r'\1', text)

    # Ensure there is one space after punctuation marks if needed (except at the end of the sentence)
    text = re.sub(r'([?.!,;:])(\S)', r'\1 \2', text)

    # Handle parentheses: no space after opening '(' or before closing ')'
    text = re.sub(r'\(\s*', '(', text)  # No spaces after '('
    text = re.sub(r'\s*\)', ')', text)  # No spaces before ')'

    # Handle curly braces: no space after opening '{' or before closing '}'
    text = re.sub(r'\{\s*', '{', text)  # No spaces after '{'
    text = re.sub(r'\s*\}', '}', text)  # No spaces before '}'

    # Handle square brackets: no space after opening '[' or before closing ']'
    text = re.sub(r'\[\s*', '[', text)  # No spaces after '['
    text = re.sub(r'\s*\]', ']', text)  # No spaces before ']'

    # Handle quotes (both single and double quotes): no spaces inside quotes
    text = re.sub(r'\s*["\']\s*', lambda match: match.group(0).strip(), text)

    # Handle vertical bar (pipe) symbol: no spaces around '|'
    text = re.sub(r'\s*\|\s*', '|', text)

    # Remove double spaces, if they appear after correction
    text = re.sub(r'\s{2,}', ' ', text)

    return text

# print(correct_punctuation_spacing("listening in the night . Listening ( in ) vain . For the"))
