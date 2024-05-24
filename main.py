from src.data_preprocessing import load_data, preprocess_data, save_data
from src.train_model import train_model, save_model
from src.evaluate_model import evaluate_model

# Step 1: Data Preprocessing
df = load_data('data/raw/reviews.csv')
df = preprocess_data(df, 'review')
save_data(df, 'data/processed/reviews_preprocessed.csv')

# Step 2: Train Model
model, vectorizer = train_model(df, 'review', 'sentiment')
save_model(model, vectorizer, 'models/sentiment_model.pkl', 'models/vectorizer.pkl')

# Step 3: Evaluate Model
evaluate_model(df, 'review', 'sentiment', model, vectorizer)
