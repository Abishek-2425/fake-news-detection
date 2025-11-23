import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_all(
    data_path="models/train_test_data.pkl",
    vectorizer_path="models/tfidf_vectorizer.pkl"
):
    """
    Load all trained models and evaluate them on the same test set.
    Prints full metrics and displays confusion matrices side-by-side.
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

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for idx, (name, path) in enumerate(model_info):

        print(f"\n==================== {name} ====================")

        model = joblib.load(path)
        y_pred = model.predict(X_test)

        # Basic metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro")
        prec_weighted = precision_score(y_test, y_pred, average="weighted")
        rec_macro = recall_score(y_test, y_pred, average="macro")
        rec_weighted = recall_score(y_test, y_pred, average="weighted")
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        # ROC-AUC — valid only for probability-capable models
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, y_score)
            else:
                roc_auc = "N/A"
        except:
            roc_auc = "N/A"

        # Print extended metrics
        print(f"Accuracy            : {acc:.4f}")
        print(f"Precision (macro)   : {prec_macro:.4f}")
        print(f"Precision (weighted): {prec_weighted:.4f}")
        print(f"Recall (macro)      : {rec_macro:.4f}")
        print(f"Recall (weighted)   : {rec_weighted:.4f}")
        print(f"F1 Score (macro)    : {f1_macro:.4f}")
        print(f"F1 Score (weighted) : {f1_weighted:.4f}")
        print(f"ROC-AUC             : {roc_auc}")
        
        # Classification report
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, digits=4))

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
