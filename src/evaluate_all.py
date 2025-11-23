import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate_all(data_path="models/train_test_data.pkl",
                 vectorizer_path="models/tfidf_vectorizer.pkl"):
    """
    Loads all trained models and evaluates them on the same test set.
    Displays comparison metrics and saves 3 confusion matrices side-by-side.
    """

    print("📂 Loading TF-IDF train/test data...")
    X_train, X_test, y_train, y_test = joblib.load(data_path)

    print("📂 Loading vectorizer...")
    vectorizer = joblib.load(vectorizer_path)

    model_info = [
        ("Logistic Regression", "models/logistic_model.pkl"),
        ("SVM (LinearSVC)", "models/svm_model.pkl"),
        ("Naive Bayes", "models/nb_model.pkl")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, path) in enumerate(model_info):
        print(f"\n============== {name} ==============")

        model = joblib.load(path)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
        axes[idx].set_title(name)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("results/confusion_matrix_all.png", dpi=300)
    print("\n📁 Saved combined confusion matrix to: results/confusion_matrix_all.png")
    plt.show()


if __name__ == "__main__":
    evaluate_all()
