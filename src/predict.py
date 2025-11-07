import os
import joblib
import pandas as pd
from src.utils import clean_text

# === Correct paths (based on your structure) ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'logistic_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'tfidf_vectorizer.pkl')

def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer safely."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"❌ TF-IDF vectorizer not found at {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer

def predict_news(title: str, text: str) -> str:
    """
    Predict whether a given news article is Fake or Real.
    Args:
        title (str): Title of the news article.
        text (str): Body text of the news article.
    Returns:
        str: 'FAKE' or 'REAL'
    """
    model, vectorizer = load_model_and_vectorizer()

    # Combine and clean text
    combined_text = clean_text(title + " " + text)
    transformed = vectorizer.transform([combined_text])

    # Predict
    prediction = model.predict(transformed)[0]
    if (prediction == 0):
        return "FAKE"
    else:
        return "REAL"


def predict_from_csv(csv_path: str):
    """
    Predicts all articles in a given CSV file.
    The CSV must contain 'title' and 'text' columns.
    """
    model, vectorizer = load_model_and_vectorizer()

    data = pd.read_csv(csv_path)
    if not {'title', 'text'}.issubset(data.columns):
        raise ValueError("CSV must contain 'title' and 'text' columns.")

    data['combined'] = data['title'] + " " + data['text']
    data['cleaned'] = data['combined'].apply(clean_text)
    X = vectorizer.transform(data['cleaned'])

    preds = model.predict(X)
    data['predicted_label'] = ["FAKE" if p == 0 else "REAL" for p in preds]

    print("\n🔍 Sample predictions:")
    print(data[['title', 'predicted_label']].head())
    return data


if __name__ == "__main__":
    print("\n🔮 Testing Fake News Prediction...\n")

    # Example usage
    sample_title = "SpaceX Successfully Launches New Batch of Starlink Satellites"
    sample_text = "This is a sample news article text used to demonstrate the prediction functionality of the fake news detection model. It contains information that may or may not be accurate, and the model will analyze it to determine its authenticity."
    result = predict_news(sample_title, sample_text)
    print(f"🧠 Prediction: {result}")
