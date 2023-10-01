# Preprocessing the data

import pandas as pd
import re
import spacy


# Load dataset
df = pd.read_csv('ready_for_preprocessing.csv')

# Function to clean text
def clean_text(text):
    # Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)

    # Remove non-alphanumeric characters and symbols
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()
    return text

# Apply the cleaning function to the review text column
df['cleaned_review'] = df['review_text'].apply(clean_text)

emoticon_mapping = {
    ':)': 'positive',
    ':(': 'negative',
    # Add more emoticons and their corresponding labels as needed
}

# Function to replace emoticons
def replace_emoticons(text):
    for emoticon, sentiment in emoticon_mapping.items():
        text = text.replace(emoticon, sentiment)
    return text

# Apply the emoticon replacement function
df['cleaned_review'] = df['cleaned_review'].apply(replace_emoticons)

# Load the spaCy language model
nlp = spacy.load('en_core_web_sm')

# Function to tokenize text and remove stopwords using spaCy
def tokenize_and_remove_stopwords(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return ' '.join(tokens)

# Apply tokenization and stopword removal
df['tokenized_review'] = df['cleaned_review'].apply(tokenize_and_remove_stopwords)

# Save the preprocessed data to a new CSV file
df.to_csv('preprocessed_amazon_fashion.csv', index=False)

# preprocessed_amazon_fashion.csv



# Conducting the sentiment analysis using VADER lexicon

import nltk
nltk.download('vader_lexicon')

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_amazon_fashion.csv')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define a function to get sentiment scores for each review
def get_sentiment(review):
    sentiment_scores = sia.polarity_scores(review)
    return sentiment_scores

# Apply the function to calculate sentiment scores for each review
df['sentiment_scores'] = df['review_text'].apply(get_sentiment)

# Extract the compound sentiment score (a normalized score)
df['compound_sentiment'] = df['sentiment_scores'].apply(lambda x: x['compound'])

# Define a function to categorize the sentiment as positive, negative, or neutral
def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply the function to categorize the sentiment
df['sentiment_category'] = df['compound_sentiment'].apply(categorize_sentiment)

# Display the DataFrame with sentiment scores and categories
print(df[['review_text', 'compound_sentiment', 'sentiment_category']])

average_sentiment = df['compound_sentiment'].mean()
print("Average Sentiment Score:", average_sentiment)

# Specify the file path
output_file_path = '/Users/marine/Desktop/py4e/sentiment_analysis_results.csv'

# Save the DataFrame to a CSV file
df.to_csv(output_file_path, index=False)  # Changed 'sentiment_analysis_results_df' to 'df'


# sentiment_analysis_results.csv
