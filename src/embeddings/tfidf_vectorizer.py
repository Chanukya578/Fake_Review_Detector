from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(max_features=5000):
    """
    Creates a TF-IDF vectorizer for text feature extraction.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) weighs terms based on their 
    frequency in a document and their rarity across the corpus, highlighting 
    distinctive terms.
    
    Args:
        max_features (int): Maximum number of features to extract
        
    Returns:
        sklearn.feature_extraction.text.TfidfVectorizer: Configured TF-IDF vectorizer
    """
    return TfidfVectorizer(
        max_features=max_features,
        # Additional parameters can be customized here
        # sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf))
        # min_df=5,          # Minimum document frequency
        # max_df=0.9,        # Maximum document frequency
        # ngram_range=(1, 2) # Use unigrams and bigrams
    )