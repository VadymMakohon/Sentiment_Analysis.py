import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def load_model(model_filepath, vectorizer_filepath):
    model = joblib.load(model_filepath)
    vectorizer = joblib.load(vectorizer_filepath)
    return model, vectorizer

def evaluate_model(df, text_column, label_column, model, vectorizer):
    X = vectorizer.transform(df[text_column])
    y_true = df[label_column]
    y_pred = model.predict(X)
    
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    df = load_data('data/processed/reviews_preprocessed.csv')
    model, vectorizer = load_model('models/sentiment_model.pkl', 'models/vectorizer.pkl')
    evaluate_model(df, 'review', 'sentiment', model, vectorizer)
