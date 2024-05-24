# Sentiment_Analysis.py

This project aims to build a sentiment analysis model using Natural Language Processing (NLP) techniques and machine learning algorithms. The model classifies text data (e.g., product reviews) as positive or negative.


## How it Works

### Data Preprocessing

**File: `src/data_preprocessing.py`**
- Loads raw data from `data/raw/reviews.csv`.
- Preprocesses the text data by converting it to lowercase, removing punctuation and stop words, and tokenizing.
- Saves the preprocessed data to `data/processed/reviews_preprocessed.csv`.

### Model Training

**File: `src/train_model.py`**
- Loads the preprocessed data.
- Vectorizes the text data using TF-IDF.
- Trains a Logistic Regression model.
- Saves the trained model and the TF-IDF vectorizer to the `models/` directory.

### Model Evaluation

**File: `src/evaluate_model.py`**
- Loads the preprocessed data, trained model, and TF-IDF vectorizer.
- Evaluates the model's performance on the preprocessed data, printing accuracy and a classification report.

### Notebooks

**File: `notebooks/Sentiment_Analysis.ipynb`**
- A Jupyter notebook that contains the exploratory data analysis (EDA), model training, and evaluation steps. This notebook provides a step-by-step guide and visualizations to understand the data and model performance.

### Main Script

**File: `main.py`**
- Orchestrates the entire pipeline by calling the data preprocessing, model training, and evaluation scripts in sequence.

### Data Files

**File: `data/raw/reviews.csv`**
- Contains the raw reviews data with columns `review` and `sentiment`.

**File: `data/processed/reviews_preprocessed.csv`**
- Contains the preprocessed reviews data after running the `data_preprocessing.py` script.

### Model Files

**File: `models/sentiment_model.pkl`**
- Contains the trained sentiment analysis model.

### Requirements

**File: `requirements.txt`**
- Lists the required Python packages to run the project.

### Documentation

**File: `README.md`**
- This file. Provides an overview of the project, how it works, usage instructions, and an example.

## Usage

To run the entire pipeline (preprocess data, train the model, and evaluate the model), execute the following command: ![Screenshot 2024-05-24 at 1 31 23 PM](https://github.com/VadymMakohon/Sentiment_Analysis.py/assets/138728243/4a5b6e61-952a-47c6-9d75-bc6c3bcf35e5)

# Example
To generate a sentiment analysis model, follow these steps:

Ensure you have a CSV file with reviews in data/raw/reviews.csv with columns review and sentiment.

Run the main.py script: $ python main.py

# Contributors
- [Vadym Makohon](https://github.com/VadymMakohon)



