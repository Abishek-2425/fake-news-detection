import os
from src.data_preprocessing import preprocess_data
from src.feature_extraction import extract_features
from src.train_model import train_logistic_model
from src.evaluate_model import evaluate_model

def main():
    print("\n🚀 Starting Fake News Detection Pipeline...\n")

    # Step 1: Data Preprocessing
    processed_data_path = "data/processed/cleaned_data.csv"
    if not os.path.exists(processed_data_path):
        print("🧹 Step 1: Preprocessing raw data...")
        preprocess_data("data/raw/Fake.csv", "data/raw/True.csv", output_dir="data/processed")
    else:
        print("✅ Preprocessed data already exists. Skipping preprocessing.")

    # Step 2: Feature Extraction
    feature_data_path = "data/processed/train_test_data.pkl"
    if not os.path.exists(feature_data_path):
        print("\n🔠 Step 2: Extracting TF-IDF features...")
        extract_features(processed_data_path, output_dir="data/processed")
    else:
        print("✅ TF-IDF features already exist. Skipping feature extraction.")

    # Step 3: Model Training
    model_path = "models/logistic_model.pkl"
    if not os.path.exists(model_path):
        print("\n⚙️ Step 3: Training Logistic Regression model...")
        train_logistic_model(data_path=feature_data_path, model_path=model_path)
    else:
        print("✅ Model already exists. Skipping training.")

    # Step 4: Evaluation
    print("\n📊 Step 4: Evaluating model performance...")
    evaluate_model(model_path=model_path, data_path=feature_data_path, output_dir="reports")

    print("\n🎯 Pipeline execution complete! Reports saved in 'reports/' directory.\n")


if __name__ == "__main__":
    main()
