import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

def extract_features(data_path, output_dir="models/", max_features=5000):
    """
    Loads cleaned data, applies TF-IDF vectorization, and splits into train/test sets.

    Parameters
    ----------
    data_path : str
        Path to the cleaned dataset (output of data_preprocessing.py)
    output_dir : str
        Directory where processed vector files will be saved
    max_features : int
        Maximum number of words to keep in the TF-IDF vocabulary

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Train/test splits of features and labels
    """
    print("📂 Loading cleaned dataset...")
    df = pd.read_csv(data_path)

    # Extract text and labels
    X = df["cleaned_text"].astype(str)
    y = df["label"]

    print("🔠 Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save vectorizer and dataset splits
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump((X_train, X_test, y_train, y_test), os.path.join(output_dir, "train_test_data.pkl"))

    print(f"✅ TF-IDF vectorizer and split data saved to {output_dir}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Local test run (optional)
    extract_features("data/processed/cleaned_data.csv")
