import os
import pandas as pd
import re
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources (only first time)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def _clean_text(text):
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


def clean_data(fake_path, true_path, output_path):
    """
    Reads Fake and True news datasets, labels them, cleans the text,
    and saves a processed dataset ready for feature extraction.
    """
    # Load datasets
    print("📥 Loading datasets...")
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    # Add labels (0 = fake, 1 = real)
    fake['label'] = 0
    true['label'] = 1

    # Combine and shuffle
    df = pd.concat([fake, true], axis=0)
    df = shuffle(df, random_state=42).reset_index(drop=True)

    # Combine title and text into one field
    df['combined_text'] = df['title'].astype(str) + " " + df['text'].astype(str)

    # Drop irrelevant columns
    df = df[['combined_text', 'subject', 'label']]

    # Clean text column
    print("🧹 Cleaning text data (this may take a few minutes)...")
    df['cleaned_text'] = df['combined_text'].apply(_clean_text)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    # Local test run (optional)
    clean_data(
        fake_path="data/raw/Fake.csv",
        true_path="data/raw/True.csv",
        output_path="data/processed/cleaned_data.csv"
    )
