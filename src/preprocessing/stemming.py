from nltk.stem import PorterStemmer

def apply_stemming(text):
    """
    Applies Porter stemming to all words in the text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with words stemmed
    """
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in str(text).split()])