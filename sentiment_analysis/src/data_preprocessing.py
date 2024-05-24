import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import string

nltk.download('punkt')
nltk.download('stopwords')

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def preprocess_data(df, text_column):
    df[text_column] = df[text_column].apply(preprocess_text)
    return df

def save_data(df, filepath):
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    df = load_data('data/raw/reviews.csv')
    df = preprocess_data(df, 'review')
    save_data(df, 'data/processed/reviews_preprocessed.csv')
