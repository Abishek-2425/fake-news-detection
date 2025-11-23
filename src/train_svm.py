import os
import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_svm_model(data_path="models/train_test_data.pkl",
                    model_path="models/svm_model.pkl"):
    """
    Trains an SVM (LinearSVC) model on the TF-IDF features and saves it.
    """

    print("📂 Loading vectorized train/test data...")
    X_train, X_test, y_train, y_test = joblib.load(data_path)

    print("⚙️ Training SVM (LinearSVC) model...")
    model = LinearSVC()
    model.fit(X_train, y_train)

    # Evaluate basic metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📊 Model Performance (SVM):")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ SVM model saved successfully to: {model_path}")

    return model


if __name__ == "__main__":
    train_svm_model()
