import argparse
import os
from data_loader import JSONDataLoader
from vectorizer.tfidf_vectorizer import TFIDFVectorizer
from model.svm_model import SVMModel
from utils.metrics import evaluate_and_save


def main(data_path: str):
    print("\n Starting DDI SVM Pipeline...")
    print("=" * 60)


    #1.Load dataset
    print(f"[1] Loading dataset from: {data_path}")
    loader = JSONDataLoader(
        train_path=os.path.join(data_path, "train.json"),
        valid_path=os.path.join(data_path, "valid.json"),
        test_path=os.path.join(data_path, "test.json")
    )

    train_texts, train_labels = loader.load_split("train")
    valid_texts, valid_labels = loader.load_split("valid")
    test_texts, test_labels = loader.load_split("test")

    print(f"[INFO] #Train={len(train_texts)}, #Valid={len(valid_texts)}, #Test={len(test_texts)}")

 
    # 2.TF-IDF Vectorization
 
    print("\n[2] Vectorizing text using TF-IDF ...")
    vectorizer = TFIDFVectorizer(max_features=50000, ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(train_texts)
    X_valid = vectorizer.transform(valid_texts)
    X_test = vectorizer.transform(test_texts)

    print(f"[INFO] Feature matrix shape: {X_train.shape}")

 
    # 3.Train SVM Model
 
    print("\n[3] Training SVM model...")
    model = SVMModel(C=1.0, max_iter=5000)

    acc, f1 = model.train(
        X_train, train_labels,
        X_valid, valid_labels
    )
    print(f"[RESULT] Train/Valid — Acc={acc:.4f}, Macro-F1={f1:.4f}")

 
    # Save model & vectorizer
 
    os.makedirs("output/svm_results", exist_ok=True)
    model.save("output/svm_results/svm_model.pkl")
    vectorizer.save("output/svm_results/tfidf.pkl")
    print("[INFO] Model & vectorizer saved.")

 
    # 4.Testing
 
    print("\n[4] Testing on held-out test set...")
    test_acc, test_f1, cm = evaluate_and_save(
        model,
        test_texts,
        X_test,
        test_labels,
        out_dir="output/svm_results"
    )
    print(f"[RESULT] Test — Acc={test_acc:.4f}, Macro-F1={test_f1:.4f}")

    print("\nDone.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM-based Drug-Drug Interaction (DDI) Prediction Pipeline")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to folder containing train/valid/test JSON files."
    )

    args = parser.parse_args()
    main(args.data_path)
