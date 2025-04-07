import nltk
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
try:
    nltk.download('wordnet', quiet=True)
except:
    pass

def apply_lemmatization(text):
    """
    Applies WordNet lemmatization to all words in the text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with words lemmatized
    """
    try:
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in str(text).split()])
    except LookupError as e:
        print(f"Warning: Could not perform lemmatization: {str(e)}")
        print("Skipping lemmatization step")
        return text