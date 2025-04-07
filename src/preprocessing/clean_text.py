import string
from nltk.corpus import stopwords
import nltk

# Ensure NLTK resources are downloaded
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

def remove_punct_and_digits(text):
    """
    Removes punctuation and digits from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with punctuation and digits removed
    """
    # First convert to string to handle any non-string inputs
    text = str(text)
    
    # Remove all punctuation characters
    for char in string.punctuation:
        text = text.replace(char, '')
    
    # Remove all digits
    text = ''.join([char for char in text if not char.isdigit()])
    
    return text

def lowercase_text(text):
    """
    Converts text to lowercase.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Lowercase text
    """
    return str(text).lower()

def remove_stopwords(text):
    """
    Removes English stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stopwords removed
    """
    try:
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in str(text).split() if word not in stop_words])
    except LookupError as e:
        print(f"Warning: Could not load stopwords: {str(e)}")
        print("Skipping stopword removal step")
        return text

def tokenize_text(text):
    """
    Splits text into tokens.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of tokens
    """
    return str(text).split()