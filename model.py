from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def train_evaluate_svm(X, y, test_size=0.2, random_state=404, kernel='linear', average='weighted'):
    """
    Trains an SVM model and evaluates the F1 score.
    
    Parameters:
    - X: Features
    - y: Target variable
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Seed used by the random number generator
    - kernel: Kernel type to be used in SVM (e.g., 'linear', 'poly', 'rbf')
    - average: Type of averaging to calculate F1 score ('micro', 'macro', 'weighted')
    
    Returns:
    - F1 score for the test set predictions
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the SVM model
    svm_model = SVC(kernel=kernel)

    # Fit the model to the training data
    svm_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_model.predict(X_test)

    # Evaluate the F1 score
    f1 = round(f1_score(y_test, y_pred, average=average),4)
    print(f"F1 Score: {f1}")
    return f1
