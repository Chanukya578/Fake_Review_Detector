import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
# import .config as cfg
from . import config as cfg

def load_model(embedding_type, classifier_type, models_dir=cfg.MODEL_DIR):
    """
    Loads a trained model based on embedding type and classifier type.
    
    Args:
        embedding_type (str): Type of embedding ('bow' or 'tfidf')
        classifier_type (str): Type of classifier ('svm', 'dt', 'rf', or 'lr')
        models_dir (str): Directory containing saved models
        
    Returns:
        object: Loaded model pipeline
    """
    # Construct model name
    model_name = f"{embedding_type}_{classifier_type}_model.pkl"
    model_path = os.path.join(models_dir, model_name)
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    # Load model
    try:
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def save_model(model, embedding_type, classifier_type, models_dir=cfg.MODEL_DIR):
    """
    Saves a trained model pipeline.
    
    Args:
        model: Model to save
        embedding_type (str): Type of embedding ('bow' or 'tfidf')
        classifier_type (str): Type of classifier ('svm', 'dt', 'rf', or 'lr')
        models_dir (str): Directory to save model
        
    Returns:
        str: Path to saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Construct model name
    model_name = f"{embedding_type}_{classifier_type}_model.pkl"
    model_path = os.path.join(models_dir, model_name)
    
    # Save model
    try:
        print(f"Saving model to {model_path}...")
        joblib.dump(model, model_path)
        print(f"Model saved successfully")
        return model_path
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}")

def split_dataset(input_file='data/raw/fake_reviews.csv', 
                  train_size=0.7, val_size=0.15, test_size=0.15, 
                  random_state=42, output_dir='data/preprocessed/'):
    """
    Splits the dataset into training, validation and test sets.
    If split files already exist, loads and returns them.
    
    Args:
        input_file (str): Path to input CSV file
        train_size (float): Proportion of data for training
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
        output_dir (str): Directory to save split files
        
    Returns:
        tuple: DataFrames for train, validation and test sets
    """
    # Define output paths
    train_path = cfg.TRAIN_PATH
    val_path = cfg.VAL_PATH
    test_path = cfg.TEST_PATH
    
    # Check if files already exist
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Split files already exist. Loading them...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print(f"Loaded training set: {len(train_df)} samples")
        print(f"Loaded validation set: {len(val_df)} samples")
        print(f"Loaded test set: {len(test_df)} samples")
        return train_df, val_df, test_df
    
    # If files don't exist, split the dataset
    print("Split files not found. Creating them...")
    
    # Check input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Verify split proportions
    if abs(train_size + val_size + test_size - 1.0) > 1e-10:
        raise ValueError("Split proportions must sum to 1")
    
    # First split: training vs. (validation + test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_size, 
        random_state=random_state, 
        stratify=df['label'] if 'label' in df.columns else None
    )
    
    # Second split: validation vs. test
    # Recalculate split ratio for validation set from the remaining data
    val_ratio = val_size / (val_size + test_size)
    
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=val_ratio, 
        random_state=random_state, 
        stratify=temp_df['label'] if 'label' in temp_df.columns else None
    )
    
    # Save split datasets
    print(f"Saving training set ({len(train_df)} samples) to {train_path}")
    train_df.to_csv(train_path, index=False)
    
    print(f"Saving validation set ({len(val_df)} samples) to {val_path}")
    val_df.to_csv(val_path, index=False)
    
    print(f"Saving test set ({len(test_df)} samples) to {test_path}")
    test_df.to_csv(test_path, index=False)
    
    print("Dataset splitting completed!")
    return train_df, val_df, test_df

def ensure_dir_exists(directory):
    """
    Creates a directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")