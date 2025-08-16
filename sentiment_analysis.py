# # sentiment_analysis.py
# from transformers import pipeline
# import pandas as pd
#
#
# def perform_sentiment_analysis(df, positive_threshold=0.5, negative_threshold=0.5):
# #Using the 'distilbert-base-uncased-finetuned-sst-2-english' model for sentiment analysis
#     classifier = pipeline('sentiment-analysis')
#
#     # Apply sentiment analysis to each message
#     results = classifier(df['message'].tolist())
#
#     # Extract sentiment labels and scores
#     sentiments = [result['label'] for result in results]
#     scores = [result['score'] for result in results]
#
#     # Add sentiment-related columns to the DataFrame
#     df['sentiment_label'] = sentiments
#     df['sentiment_score'] = scores
#
#     return df

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def perform_sentiment_analysis(df, positive_threshold=0.9):
    # Use the 'distilbert-base-uncased-finetuned-sst-2-english' model for sentiment analysis
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    # Load the model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Define the sentiment analysis function
    def analyze_sentiment(message):
        inputs = tokenizer(message, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = logits.softmax(dim=1)
        return probabilities

    # Apply sentiment analysis to each message
    results = df['message'].apply(analyze_sentiment)

    # Initialize empty lists for positive and negative probabilities
    positive_probs = []
    negative_probs = []

    for result in results:
        # Extract positive and negative probabilities
        positive_prob = result[0][1].item()
        negative_prob = result[0][0].item()

        # Append probabilities to the respective lists
        positive_probs.append(positive_prob)
        negative_probs.append(negative_prob)

    # Add sentiment-related columns to the DataFrame
    df['sentiment_label'] = ["positive" if p >= positive_threshold else "negative" for p in positive_probs]
    df['sentiment_score'] = [max(p, n) for p, n in zip(positive_probs, negative_probs)]

    return df
