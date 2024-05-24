import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def train_model(df, text_column, label_column):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[text_column])
    y = df[label_column]
    
    model = LogisticRegression()
    model.fit(X, y)
    
    return model, vectorizer

def save_model(model, vectorizer, model_filepath, vectorizer_filepath):
    joblib.dump(model, model_filepath)
    joblib.dump(vectorizer, vectorizer_filepath)

if __name__ == "__main__":
    df = load_data('data/processed/reviews_preprocessed.csv')
    model, vectorizer = train_model(df, 'review', 'sentiment')
    save_model(model, vectorizer, 'models/sentiment_model.pkl', 'models/vectorizer.pkl')
