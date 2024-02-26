import pandas as pd
import re
import spacy

# Load dataset
df = pd.read_csv('aggregated_data.csv')

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
df.to_csv('processed_fashion_reviews.csv', index=False)
