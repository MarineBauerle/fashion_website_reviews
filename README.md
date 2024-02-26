# Fashion Reviews - Sentiment Analysis

## Project Description

This project undertakes a comprehensive sentiment analysis of fashion product reviews, aiming to extract and understand the underlying emotions and opinions of customers. The goal is to generate actionable insights that businesses can use to enhance their products, customer service, and overall customer experience.

## Methodology

### 1. Data Collection

The raw dataset containing fashion reviews is located in the `data/raw/` directory as `Original_sample.csv`.

### 2. Data Preprocessing

#### Raw to Processed Data:

Data preprocessing is a foundational step to ensure the quality of the analysis. The steps involve:

- **Text Cleaning**: Removal of any unwanted characters, URLs, numbers, and standardizing text data.
- **Handling Missing Values**: Identifying and addressing any null or missing values in the dataset.
- **Text Transformation**: Tokenization, stemming, and lemmatization to break down the reviews and reduce words to their base forms.

Processed data files can be found in the `data/processed/` directory:

- `processed_amazon_fashion.csv`: Contains cleaned and structured reviews.
- `aggregated_data.csv`: Grouped data based on certain criteria (e.g., product type, brand).

### 3. Sentiment Analysis

Utilizing the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool, each review was evaluated for its sentiment score. VADER is particularly effective for sentiment analysis on social media and short texts, as it can understand the context of certain words and nuances like capitalization and punctuation.

The sentiment scores were categorized as:

- **Positive**
- **Negative**
- **Neutral**

The results of the sentiment analysis can be found in:

- `results_sentiment_analysis.csv` in the `data/processed/` directory.

### 4. Visualization

Key insights and patterns derived from the sentiment analysis were visualized using Python's visualization libraries and exported as a Tableau dashboard. This allows for an interactive exploration of the data, facilitating a deeper understanding of customer sentiments across various product categories.

Visualizations can be found in the `results/` directory:

- `Tableau sentiment analysis.pdf`: A snapshot of the Tableau dashboard.
- `overall count results.pdf`: Graphical representation of overall sentiment scores.

### 5. Analysis & Conclusions

From the sentiment analysis, one can conclude about:

- Overall customer satisfaction levels for the website's fashion products.
- Specific product categories or brands that might require improvements.
- Trends in customer sentiments over time or across different product launches.

## Dataset

You can access the dataset used in this project [Fashion Review Data](https://nijianmo.github.io/amazon/index.html)

In this project, I have used the subset named 'Amazon fashion'.
