from sklearn.svm import SVC

def get_svm_classifier(kernel='rbf', C=1.0, probability=True, random_state=42):
    """
    Creates a Support Vector Machine classifier.
    
    SVM finds the hyperplane that best separates classes in high-dimensional space,
    making it effective for text classification with high-dimensional feature vectors.
    
    Args:
        kernel (str): Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
        C (float): Regularization parameter
        probability (bool): Whether to enable probability estimates
        random_state (int): Random number generator seed
        
    Returns:
        sklearn.svm.SVC: Configured SVM classifier
    """
    return SVC(
        kernel=kernel,
        C=C,
        probability=probability,
        random_state=random_state
    )