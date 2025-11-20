import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    """
    Cleans text by removing URLs, punctuation, numbers, and stopwords.
    Also lowercases and lemmatizes the words.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Remove URLs, non-alphabets, and extra spaces
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()

    # Lemmatize and remove stopwords
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

