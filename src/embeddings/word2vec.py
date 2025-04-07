import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def get_bow_vectorizer(max_features=5000):
    """
    Creates a Bag of Words (BoW) vectorizer for text feature extraction.
    
    BoW represents text as the frequency of each word in the document,
    ignoring word order but capturing word occurrence patterns.
    
    Args:
        max_features (int): Maximum number of features to extract
        
    Returns:
        sklearn.feature_extraction.text.CountVectorizer: Configured BoW vectorizer
    """
    return CountVectorizer(
        max_features=max_features,
        # Additional parameters can be customized here
        # min_df=5,          # Minimum document frequency  
        # max_df=0.9,        # Maximum document frequency
        # ngram_range=(1, 2) # Use unigrams and bigrams
    )