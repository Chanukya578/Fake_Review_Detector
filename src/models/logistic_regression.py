from sklearn.linear_model import LogisticRegression

def get_logistic_regression_classifier(C=1.0, solver='liblinear', max_iter=1000, random_state=42):
    """
    Creates a Logistic Regression classifier.
    
    Logistic Regression models the probability of a binary outcome, 
    and is effective for text classification especially with high-dimensional data.
    
    Args:
        C (float): Inverse of regularization strength
        solver (str): Algorithm to use in optimization
        max_iter (int): Maximum number of iterations
        random_state (int): Random number generator seed
        
    Returns:
        sklearn.linear_model.LogisticRegression: Configured Logistic Regression classifier
    """
    return LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )