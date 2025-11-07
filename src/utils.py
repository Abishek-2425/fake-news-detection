import re
import string

def clean_text(text):
    """
    Basic text cleaning function.
    - Lowercases
    - Removes punctuation and numbers
    - Removes extra whitespace
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
