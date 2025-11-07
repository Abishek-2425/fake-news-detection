import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_logistic_model(data_path="data/processed/train_test_data.pkl",
                         model_path="models/logistic_model.pkl"):
    """
    Trains a Logistic Regression model on the TF-IDF features and saves it.

    Parameters
    ----------
    data_path : str
        Path to the train/test TF-IDF data file saved by feature_extraction.py
    model_path : str
        Path to save the trained logistic regression model

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        The trained model
    """
    print("📂 Loading vectorized train/test data...")
    X_train, X_test, y_train, y_test = joblib.load(data_path)

    # Create and train logistic regression model
    print("⚙️ Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200, solver='lbfgs', n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate basic metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📊 Model Performance:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save trained model
    joblib.dump(model, model_path)
    print(f"✅ Model saved successfully to: {model_path}")

    return model


if __name__ == "__main__":
    # Local test run (optional)
    train_logistic_model()
