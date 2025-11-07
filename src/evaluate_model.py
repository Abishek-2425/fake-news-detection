# src/evaluate_model.py
import joblib as jb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model():
    # Load saved model and data
    model = jb.load("./models/logistic_model.pkl")
    X_train, X_test, y_train, y_test = jb.load("./models/train_test_data.pkl")
    vectorizer = jb.load("./models/tfidf_vectorizer.pkl")

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("✅ Model Evaluation Results")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-Score : {f1:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
    disp.plot(cmap='cool')
    plt.title("Confusion Matrix")
    plt.savefig("./results/confusion_matrix.png")
    plt.show()

    from sklearn.metrics import roc_curve, auc
    import numpy as np

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.savefig('./results/roc_curve.png')
    plt.show()


if __name__ == "__main__":
    evaluate_model()
