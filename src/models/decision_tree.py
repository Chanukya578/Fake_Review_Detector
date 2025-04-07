from sklearn.tree import DecisionTreeClassifier

def get_decision_tree_classifier(max_depth=None, min_samples_split=2, random_state=42):
    """
    Creates a Decision Tree classifier.
    
    Decision Trees split data recursively based on features to create classification rules,
    making them interpretable and capable of capturing non-linear patterns.
    
    Args:
        max_depth (int, optional): Maximum depth of the tree
        min_samples_split (int): Minimum samples required to split a node
        random_state (int): Random number generator seed
        
    Returns:
        sklearn.tree.DecisionTreeClassifier: Configured Decision Tree classifier
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )