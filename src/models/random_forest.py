from sklearn.ensemble import RandomForestClassifier

def get_random_forest_classifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
    """
    Creates a Random Forest classifier.
    
    Random Forests combine multiple decision trees to improve accuracy and reduce overfitting,
    making them robust and versatile for text classification.
    
    Args:
        n_estimators (int): Number of trees in the forest
        max_depth (int, optional): Maximum depth of the trees
        min_samples_split (int): Minimum samples required to split a node
        random_state (int): Random number generator seed
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Configured Random Forest classifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )