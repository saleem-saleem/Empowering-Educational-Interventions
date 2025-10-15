# Fuzzy Neuro J48 (OPTIFUZZTREE) simplified implementation
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# ---------- Step 1: Fuzzification ----------
def fuzzify_features(X, num_membership=3):
    """
    Apply Gaussian membership functions to each feature
    Returns a list of membership matrices for each feature
    """
    X_scaled = MinMaxScaler().fit_transform(X)
    n_samples, n_features = X_scaled.shape
    
    membership_matrices = []
    centers = np.linspace(0, 1, num_membership)
    sigma = 0.1
    
    for i in range(n_features):
        feature_values = X_scaled[:, i]
        memberships = np.zeros((n_samples, num_membership))
        for j, c in enumerate(centers):
            memberships[:, j] = np.exp(-0.5 * ((feature_values - c)/sigma)**2)
        membership_matrices.append(memberships)
    
    return membership_matrices, centers

# ---------- Step 2: Train J48 Decision Tree ----------
def train_j48_tree(X_fuzzy, y):
    """
    Train a decision tree (J48) on fuzzy values
    Flatten the fuzzy memberships to feed into the tree
    """
    # Concatenate all membership values as new features
    X_flattened = np.hstack(X_fuzzy)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_flattened, y)
    return clf

# ---------- Step 3: Fuzzy Classification ----------
def classify_optifuzztree(X_test, X_fuzzy_train, clf, num_membership=3):
    """
    Classify new instances using fuzzy J48 approach
    """
    X_scaled = MinMaxScaler().fit_transform(X_test)
    n_samples, n_features = X_scaled.shape
    membership_matrices = []
    centers = np.linspace(0, 1, num_membership)
    sigma = 0.1
    
    # Fuzzify test instances
    for i in range(n_features):
        feature_values = X_scaled[:, i]
        memberships = np.zeros((n_samples, num_membership))
        for j, c in enumerate(centers):
            memberships[:, j] = np.exp(-0.5 * ((feature_values - c)/sigma)**2)
        membership_matrices.append(memberships)
    
    X_test_flattened = np.hstack(membership_matrices)
    
    # Predict classes using the trained decision tree
    predictions = clf.predict(X_test_flattened)
    
    # Compute fuzzy membership for each class
    classes = clf.classes_
    membership_values = np.zeros((n_samples, len(classes)))
    
    # Using leaf node probabilities for fuzzy membership approximation
    leaf_indices = clf.apply(X_test_flattened)
    tree = clf.tree_
    
    for i, leaf in enumerate(leaf_indices):
        # Retrieve class distribution at the leaf
        distribution = tree.value[leaf][0]
        membership_values[i, :] = distribution / np.sum(distribution)
    
    # Predicted class with highest membership
    predicted_classes = classes[np.argmax(membership_values, axis=1)]
    
    return predicted_classes, membership_values

# ---------- Example Usage ----------
if __name__ == "__main__":
    # Create synthetic dataset
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                               n_redundant=1, n_classes=2, random_state=42)
    
    # Step 1: Fuzzify features
    X_fuzzy, centers = fuzzify_features(X, num_membership=3)
    
    # Step 2: Train fuzzy J48 (OPTIFUZZTREE)
    clf = train_j48_tree(X_fuzzy, y)
    
    # Step 3: Classify using fuzzy J48
    predictions, membership_values = classify_optifuzztree(X, X_fuzzy, clf, num_membership=3)
    
    # Evaluate accuracy
    accuracy = np.mean(predictions == y)
    print("Predicted classes:", predictions)
    print("True classes:", y)
    print("Classification Accuracy:", accuracy)
    print("Fuzzy membership values (first 5 samples):\n", membership_values[:5])
