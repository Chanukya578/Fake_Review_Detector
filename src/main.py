import os
import pandas as pd
import argparse
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from . import config as cfg

# Import local modules
from utils import split_dataset, save_model, load_model, ensure_dir_exists
from preprocessing_DONE.preprocessing_pipeline import load_and_preprocess_dataset, preprocess_text

def preprocess_datasets(raw_data_dir='data/raw', 
                        processed_data_dir='data/preprocessed'):
    """
    Preprocesses the train, validation and test datasets.
    
    Args:
        raw_data_dir (str): Directory containing raw data files
        processed_data_dir (str): Directory to save preprocessed files
        
    Returns:
        tuple: Paths to preprocessed train, validation and test files
    """
    # Ensure directories exist
    ensure_dir_exists(processed_data_dir)
    
    # Define file paths
    train_path = cfg.TRAIN_PATH
    val_path = cfg.VAL_PATH
    test_path = cfg.TEST_PATH
    
    # Check if raw files exist
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("Raw split datasets not found. Splitting the dataset first...")
        train_path, val_path, test_path = split_dataset(
            input_file=os.path.join(raw_data_dir, 'fake_reviews.csv'),
            output_dir=raw_data_dir
        )
    
    # Define output paths
    preprocessed_train_path = cfg.PREPROCESSED_TRAIN_PATH
    preprocessed_val_path = cfg.PREPROCESSED_VAL_PATH
    preprocessed_test_path = cfg.PREPROCESSED_TEST_PATH
    
    # Check if preprocessed files already exist
    if (os.path.exists(preprocessed_train_path) and 
        os.path.exists(preprocessed_val_path) and 
        os.path.exists(preprocessed_test_path)):
        print("Preprocessed files already exist. Skipping preprocessing...")
        return preprocessed_train_path, preprocessed_val_path, preprocessed_test_path
    
    # Preprocess each dataset
    print("\nPreprocessing training dataset...")
    train_df = load_and_preprocess_dataset(train_path)
    train_df.to_csv(preprocessed_train_path, index=False)
    print(f"Preprocessed training data saved to {preprocessed_train_path}")
    
    print("\nPreprocessing validation dataset...")
    val_df = load_and_preprocess_dataset(val_path)
    val_df.to_csv(preprocessed_val_path, index=False)
    print(f"Preprocessed validation data saved to {preprocessed_val_path}")
    
    print("\nPreprocessing test dataset...")
    test_df = load_and_preprocess_dataset(test_path)
    test_df.to_csv(preprocessed_test_path, index=False)
    print(f"Preprocessed test data saved to {preprocessed_test_path}")
    
    return preprocessed_train_path, preprocessed_val_path, preprocessed_test_path

def create_model_pipeline(embedding_type, classifier_type):
    """
    Creates a model pipeline with the specified embedding and classifier.
    
    Args:
        embedding_type (str): Type of embedding ('bow' or 'tfidf')
        classifier_type (str): Type of classifier ('svm', 'dt', 'rf', or 'lr')
        
    Returns:
        sklearn.pipeline.Pipeline: Model pipeline
    """
    # Configure embedding method
    if embedding_type.lower() == 'bow':
        vectorizer = CountVectorizer(max_features=5000)
    else:
        raise ValueError("embedding_type must be 'bow'")
    
    # Configure classifier
    if classifier_type.lower() == 'dt':
        classifier = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("classifier_type must be 'svm', 'dt', 'rf', or 'lr'")
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    return pipeline

def train_model(embedding_type, classifier_type, train_data_path=cfg.PREPROCESSED_TRAIN_PATH, models_dir=cfg.MODEL_DIR):
    """
    Trains a model with the specified embedding and classifier on the training data.
    
    Args:
        embedding_type (str): Type of embedding ('bow' or 'tfidf')
        classifier_type (str): Type of classifier ('svm', 'dt', 'rf', or 'lr')
        train_data_path (str): Path to preprocessed training data
        models_dir (str): Directory to save the trained model
        
    Returns:
        sklearn.pipeline.Pipeline: Trained model pipeline
    """
    # Ensure models directory exists
    ensure_dir_exists(models_dir)
    
    print(f"Training {embedding_type}_{classifier_type} model...")
    
    # Load training data
    train_df = pd.read_csv(train_data_path)
    
    # Prepare training data
    X_train = train_df['text_']
    y_train = train_df["label"].map({"OR": 0, "CG": 1})
    
    # Create model pipeline
    pipeline = create_model_pipeline(embedding_type, classifier_type)
    
    # Train model
    print(f"Training model on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)
    
    # Save model
    save_model(pipeline, embedding_type, classifier_type, models_dir)
    
    return pipeline

def evaluate_model(model, eval_data_path, model_name=None):
    """
    Evaluates a model on the evaluation data.
    
    Args:
        model: Trained model pipeline
        eval_data_path (str): Path to evaluation data
        model_name (str, optional): Name of the model for reporting
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"Evaluating model on {eval_data_path}...")
    
    # Load evaluation data
    eval_df = pd.read_csv(eval_data_path)
    
    # Prepare evaluation data
    X_eval = eval_df['text_']
    y_eval = eval_df["label"].map({"OR": 0, "CG": 1})
    
    # Make predictions
    eval_probs = model.predict_proba(X_eval)[:, 1]
    eval_preds = (eval_probs >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_eval, eval_preds)
    report = classification_report(y_eval, eval_preds, output_dict=True)
    roc_auc = roc_auc_score(y_eval, eval_probs)
    
    # Print results
    print(f"Model: {model_name if model_name else 'Unknown'}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_eval, eval_preds))
    
    # Return metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'report': report,
        'predictions': eval_preds,
        'probabilities': eval_probs
    }
    
    return metrics

def train_all_models(train_data_path, models_dir=cfg.MODEL_DIR):
    """
    Trains all combinations of embedding types and classifier types.
    
    Args:
        train_data_path (str): Path to preprocessed training data
        models_dir (str): Directory to save trained models
        
    Returns:
        dict: Dictionary of trained models
    """
    embedding_types = ['bow']
    classifier_types = ['dt']
    
    trained_models = {}
    
    for embedding in embedding_types:
        for classifier in classifier_types:
            model_name = f"{embedding}_{classifier}"
            print(f"\n{'='*20} Training {model_name} {'='*20}")
            
            try:
                model = train_model(embedding, classifier, train_data_path, models_dir)
                trained_models[model_name] = model
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
    
    return trained_models

def evaluate_all_models(eval_data_path, models_dir='models'):
    """
    Evaluates all trained models on the evaluation data.
    
    Args:
        eval_data_path (str): Path to evaluation data
        models_dir (str): Directory containing trained models
        
    Returns:
        dict: Dictionary of evaluation results for each model
    """
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found.")
        return {}
    
    # Get all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
    
    if not model_files:
        print(f"Error: No models found in '{models_dir}'.")
        return {}
    
    # Evaluate each model
    eval_results = {}
    
    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '')
        print(f"\n{'='*20} Evaluating {model_name} {'='*20}")
        
        try:
            # Load model
            model_path = os.path.join(models_dir, model_file)
            model = joblib.load(model_path)
            
            # Evaluate model
            metrics = evaluate_model(model, eval_data_path, model_name)
            eval_results[model_name] = metrics
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
    
    # Find and print the best model
    if eval_results:
        best_model = max(eval_results.items(), key=lambda x: x[1]['accuracy'])
        print("\n=== Best Model Based on Accuracy ===")
        print(f"Model: {best_model[0]}")
        print(f"Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"ROC-AUC: {best_model[1]['roc_auc']:.4f}")
    
    return eval_results

def main():
    parser = argparse.ArgumentParser(description='Fake Review Detector')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'evaluate', 'train_and_evaluate'], 
                        default='train_and_evaluate', help='Mode of operation')
    parser.add_argument('--embedding', choices=['bow', 'tfidf'], default='tfidf', 
                        help='Embedding type')
    parser.add_argument('--classifier', choices=['svm', 'dt', 'rf', 'lr'], default='lr', 
                        help='Classifier type')
    parser.add_argument('--raw-data-dir', default='data/raw', 
                        help='Directory containing raw data')
    parser.add_argument('--processed-data-dir', default='data/preprocessed', 
                        help='Directory for preprocessed data')
    parser.add_argument('--models-dir', default='models', 
                        help='Directory for saving/loading models')
    parser.add_argument('--all-models', action='store_true', 
                        help='Train or evaluate all model combinations')
    
    args = parser.parse_args()
    
    if args.mode in ['preprocess', 'train', 'train_and_evaluate']:
        # Preprocess datasets
        train_path, val_path, test_path = preprocess_datasets(
            args.raw_data_dir, args.processed_data_dir
        )
    
    if args.mode in ['train', 'train_and_evaluate']:
        # Train models
        if args.all_models:
            trained_models = train_all_models(
                os.path.join(args.processed_data_dir, 'preprocessed_train.csv'),
                args.models_dir
            )
        else:
            model = train_model(
                args.embedding, args.classifier,
                os.path.join(args.processed_data_dir, 'preprocessed_train.csv'),
                args.models_dir
            )
    
    if args.mode in ['evaluate', 'train_and_evaluate']:
        # Evaluate models
        if args.all_models:
            eval_results = evaluate_all_models(
                os.path.join(args.processed_data_dir, 'preprocessed_val.csv'),
                args.models_dir
            )
        else:
            # Load model if in evaluate-only mode
            if args.mode == 'evaluate':
                model = load_model(args.embedding, args.classifier, args.models_dir)
            
            # Evaluate model
            metrics = evaluate_model(
                model,
                os.path.join(args.processed_data_dir, 'preprocessed_val.csv'),
                f"{args.embedding}_{args.classifier}"
            )

if __name__ == "__main__":
    main()