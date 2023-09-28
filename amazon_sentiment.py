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
output_file_path = '/Users/marine/Desktop/py4e/sentiment_results.csv'

# Save the DataFrame to a CSV file
df.to_csv(output_file_path, index=False)  # Changed 'sentiment_results_df' to 'df'
