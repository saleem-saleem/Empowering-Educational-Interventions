# Fuzzy Neuro PRISM (COGNIFUZZ) simplified implementation
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

# ---------- Step 1: Fuzzification ----------
def fuzzify_gaussian(X, num_membership=3):
    """
    Apply Gaussian membership functions to each feature
    Returns a list of membership matrices for each feature
    """
    X_scaled = MinMaxScaler().fit_transform(X)
    n_samples, n_features = X_scaled.shape
    
    membership_matrices = []
    
    for i in range(n_features):
        feature_values = X_scaled[:, i]
        # Create Gaussian membership centers equally spaced
        centers = np.linspace(0, 1, num_membership)
        sigma = 0.1  # standard deviation for Gaussian function
        memberships = np.zeros((n_samples, num_membership))
        
        for j, c in enumerate(centers):
            memberships[:, j] = np.exp(-0.5 * ((feature_values - c)/sigma)**2)
        membership_matrices.append(memberships)
    
    return membership_matrices, centers

# ---------- Step 2: Generate Fuzzy Rules (Simplified PRISM) ----------
def generate_fuzzy_rules(X_membership, y, threshold=0.5):
    """
    Generate fuzzy rules:
    Each rule: IF feature_i is in fuzzy set THEN class = y
    """
    n_samples = X_membership[0].shape[0]
    n_features = len(X_membership)
    rules = []
    
    # For each sample, create a rule based on max membership for each feature
    for i in range(n_samples):
        antecedent = []
        for j in range(n_features):
            fuzzy_set_idx = np.argmax(X_membership[j][i])
            antecedent.append(fuzzy_set_idx)
        consequent = y[i]
        rules.append((antecedent, consequent))
    
    return rules

# ---------- Step 3: Fuzzy Classification ----------
def classify_fuzzy(X_test, X_membership, rules):
    """
    Classify new instances using fuzzy rules
    """
    n_samples = X_test.shape[0]
    n_features = X_test.shape[1]
    classes = list(set([rule[1] for rule in rules]))
    
    # Fuzzify test instance
    test_membership = []
    for j in range(n_features):
        feature_values = X_test[:, j]
        num_membership = X_membership[j].shape[1]
        centers = np.linspace(0, 1, num_membership)
        sigma = 0.1
        memberships = np.zeros((len(feature_values), num_membership))
        for k, c in enumerate(centers):
            memberships[:, k] = np.exp(-0.5 * ((feature_values - c)/sigma)**2)
        test_membership.append(memberships)
    
    predictions = []
    
    for i in range(n_samples):
        # Compute firing strengths for all rules
        firing_strengths = []
        for rule in rules:
            antecedent, consequent = rule
            strength = 1.0
            for j in range(n_features):
                strength *= test_membership[j][i, antecedent[j]]
            firing_strengths.append((strength, consequent))
        
        # Normalize firing strengths
        total_strength = sum([f[0] for f in firing_strengths]) + 1e-6
        normalized_strengths = [(f[0]/total_strength, f[1]) for f in firing_strengths]
        
        # Compute class membership M_i
        class_membership = {c: 0.0 for c in classes}
        for strength, c in normalized_strengths:
            class_membership[c] += strength
        
        # Assign class with highest membership
        predicted_class = max(class_membership, key=class_membership.get)
        predictions.append(predicted_class)
    
    return np.array(predictions)

# ---------- Example Usage ----------
if __name__ == "__main__":
    # Create synthetic dataset
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=2, random_state=42)
    
    # Step 1: Fuzzify features
    X_membership, centers = fuzzify_gaussian(X, num_membership=3)
    
    # Step 2: Generate fuzzy rules
    rules = generate_fuzzy_rules(X_membership, y)
    
    # Step 3: Classify new instances (using training set as test for demo)
    predictions = classify_fuzzy(X, X_membership, rules)
    
    # Evaluate accuracy
    accuracy = np.mean(predictions == y)
    print("Predicted classes:", predictions)
    print("True classes:", y)
    print("Classification Accuracy:", accuracy)
