import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

# ----------------------------
# 1. Fuzzification Functions
# ----------------------------
def fuzzify_features(X, num_membership=3):
    """
    Fuzzify input features using Gaussian membership functions.
    
    Parameters:
        X : ndarray
            Input dataset of shape (n_samples, n_features)
        num_membership : int
            Number of fuzzy membership functions per feature

    Returns:
        membership_matrices : list of ndarray
            List of fuzzified matrices for each feature
        centers : ndarray
            Centers of Gaussian membership functions
    """
    X_scaled = MinMaxScaler().fit_transform(X)
    n_samples, n_features = X_scaled.shape
    membership_matrices = []
    centers = np.linspace(0, 1, num_membership)
    sigma = 0.1  # Standard deviation for Gaussian function

    for i in range(n_features):
        feature_values = X_scaled[:, i]
        memberships = np.zeros((n_samples, num_membership))
        for j, c in enumerate(centers):
            memberships[:, j] = np.exp(-0.5 * ((feature_values - c) / sigma) ** 2)
        membership_matrices.append(memberships)

    return membership_matrices, centers

# ----------------------------
# 2. Performance Metrics
# ----------------------------
def performance_metrics(y_true, y_pred):
    """
    Compute evaluation metrics: Accuracy, True Positive Rate, True Negative Rate,
    Recall, and Specificity.

    Parameters:
        y_true : array-like
            Ground truth labels
        y_pred : array-like
            Predicted labels

    Returns:
        metrics_dict : dict
            Dictionary containing metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    tpr = recall
    tnr = specificity
    
    metrics_dict = {
        'Accuracy': round(accuracy, 4),
        'Recall (TPR)': round(tpr, 4),
        'Specificity (TNR)': round(tnr, 4),
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }
    return metrics_dict

# ----------------------------
# 3. Dataset Preprocessing
# ----------------------------
def preprocess_dataset(file_path, target_column, encode_categorical=True, normalize=True):
    """
    Load and preprocess dataset:
        - Handles missing values
        - Encodes categorical variables
        - Normalizes numerical features

    Parameters:
        file_path : str
            Path to CSV dataset
        target_column : str
            Name of the target column
        encode_categorical : bool
            Whether to encode categorical features
        normalize : bool
            Whether to normalize numeric features

    Returns:
        X : ndarray
            Preprocessed feature matrix
        y : ndarray
            Target labels
    """
    df = pd.read_csv(file_path)
    
    # Fill missing values with mean for numeric, mode for categorical
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    
    # Encode categorical variables
    if encode_categorical:
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
    
    # Normalize numeric features
    if normalize:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = MinMaxScaler().fit_transform(X[numeric_cols])
    
    return X.values, y

# ----------------------------
# 4. One-Hot Encoding (Optional)
# ----------------------------
def one_hot_encode_labels(y):
    """
    Convert class labels to one-hot encoded format.
    
    Parameters:
        y : array-like
            Class labels
    
    Returns:
        y_encoded : ndarray
            One-hot encoded labels
    """
    y = np.array(y).reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y)
    return y_encoded
