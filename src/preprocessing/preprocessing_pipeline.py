import pandas as pd
import nltk
from .clean_text import lowercase_text, remove_punct_and_digits, remove_stopwords, tokenize_text
from .stemming import apply_stemming
from .lemmatization import apply_lemmatization

# Ensure all required NLTK resources are downloaded
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

def preprocess_text(df, text_column='text_'):
    """
    Applies the complete preprocessing pipeline to text data in a DataFrame.
    
    Pipeline steps:
    1. Lowercasing
    2. Removing punctuation and digits
    3. Removing stopwords
    4. Tokenization
    5. Stemming
    6. Lemmatization
    
    Args:
        df (pandas.DataFrame): DataFrame containing text data
        text_column (str): Name of the column containing text to preprocess
        
    Returns:
        pandas.DataFrame: DataFrame with preprocessed text
    """
    # Work on a copy to avoid modifying the original
    df = df.copy()
    
    # Only process if the text column exists
    if text_column in df.columns:
        print("Starting text preprocessing pipeline...")
        
        # 1. Lowercasing
        print("Applying lowercasing...")
        df[text_column] = df[text_column].apply(lowercase_text)
        
        # 2. Removing Special characters, Punctuation & Digits
        print("Removing punctuation and digits...")
        df[text_column] = df[text_column].apply(remove_punct_and_digits)
        
        # 3. Domain-specific Stopword Removal
        print("Removing stopwords...")
        df[text_column] = df[text_column].apply(remove_stopwords)
        
        # 4. Tokenization - Split text into list of tokens
        print("Tokenizing text...")
        df[f'{text_column}_tokens'] = df[text_column].apply(tokenize_text)
        
        # 5. Stemming
        print("Applying stemming...")
        df[text_column] = df[text_column].apply(apply_stemming)
        
        # 6. Lemmatization
        print("Applying lemmatization...")
        df[text_column] = df[text_column].apply(apply_lemmatization)
        
        print("Text preprocessing completed!")
    else:
        print(f"Warning: Text column '{text_column}' not found in DataFrame.")
    
    return df

def load_and_preprocess_dataset(dataset_path, text_column='text_'):
    """
    Loads a dataset from a CSV file and applies the complete preprocessing pipeline.
    
    Args:
        dataset_path (str): Path to the CSV file
        text_column (str): Name of the column containing text to preprocess
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
        
        # Drop unnecessary columns if they exist
        if 'category' in df.columns and 'rating' in df.columns:
            df.drop(columns=['category', 'rating'], inplace=True)
            print("Dropped unnecessary columns 'category' and 'rating'")
        
        # Apply preprocessing
        print(f"Preprocessing dataset using column '{text_column}'...")
        processed_df = preprocess_text(df, text_column)
        
        print(f"Dataset loaded and preprocessed successfully! Shape: {processed_df.shape}")
        return processed_df
    
    except Exception as e:
        print(f"Error loading or preprocessing dataset: {str(e)}")
        raise