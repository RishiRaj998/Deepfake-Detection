import os
import joblib

def load_model_and_vectorizer(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    saved = joblib.load(model_path)
    return saved['model'], saved['vectorizer']

def preprocess_text(text):
    return ''.join(c.lower() if c.isalpha() or c.isspace() else '' for c in text)

def predict_text(texts, model, vectorizer):
    texts_clean = [preprocess_text(text) for text in texts]
    X = vectorizer.transform(texts_clean)
    preds = model.predict(X)
    return ["REAL" if p == 1 else "FAKE" for p in preds]

def interactive_test(rf_model_path, xgb_model_path):
    rf_model, rf_vectorizer = load_model_and_vectorizer(rf_model_path)
    xgb_model, xgb_vectorizer = load_model_and_vectorizer(xgb_model_path)

    print("Enter news text to classify (type 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.strip().lower() == 'exit':
            break
        if not user_input.strip():
            print("Please enter some text or type 'exit' to quit.")
            continue

        texts = [user_input.strip()]
        rf_pred = predict_text(texts, rf_model, rf_vectorizer)[0]
        xgb_pred = predict_text(texts, xgb_model, xgb_vectorizer)[0]

        print(f"Random Forest prediction: {rf_pred}")
        print(f"XGBoost prediction: {xgb_pred}\n")

if __name__ == "__main__":
    rf_model_path = r"D:/Major Project/Rishi_Raj/text_detection/Rishi01.joblib"
    xgb_model_path = r"D:/Major Project/Rishi_Raj/text_detection/Rishi02.joblib"
    interactive_test(rf_model_path, xgb_model_path)