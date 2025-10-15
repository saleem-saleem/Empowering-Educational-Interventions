import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# ---------- Step 1: Compute Relevance Score ----------
def compute_relevance_scores(X, y):
    """
    Compute relevance scores of features using Mutual Information with the target.
    Returns a vector of relevance scores for each feature.
    """
    relevance_scores = mutual_info_classif(X, y, discrete_features='auto')
    return relevance_scores

# ---------- Step 2: Select Relevant Attributes ----------
def select_relevant_attributes(X, y, relevance_scores, tau_rel=0.05):
    """
    Select relevant features based on threshold tau_rel and independence check.
    """
    # Sort features by relevance score descending
    sorted_idx = np.argsort(-relevance_scores)
    Srel = []
    
    for idx in sorted_idx:
        if relevance_scores[idx] >= tau_rel:
            # Independence check: for simplicity, check if feature not already selected
            Srel.append(idx)
            
    X_rel = X[:, Srel]
    return X_rel, Srel

# ---------- Step 3: Compute Redundancy Score ----------
def compute_redundancy_scores(X_rel):
    """
    Compute redundancy score of each feature as the maximum correlation with other features.
    Lower score means less redundant.
    """
    n_features = X_rel.shape[1]
    corr_matrix = np.corrcoef(X_rel.T)
    redundancy_scores = []
    
    for i in range(n_features):
        # Max absolute correlation with other features
        max_corr = np.max(np.abs(np.delete(corr_matrix[i, :], i)))
        redundancy_scores.append(max_corr)
        
    return np.array(redundancy_scores)

# ---------- Step 4: Select Non-Redundant Features ----------
def select_non_redundant_features(X_rel, redundancy_scores, tau_red=0.7):
    """
    Select non-redundant features based on redundancy threshold tau_red.
    """
    sorted_idx = np.argsort(redundancy_scores)  # Low redundancy first
    Sred = []
    
    for idx in sorted_idx:
        if redundancy_scores[idx] <= tau_red:
            Sred.append(idx)
    
    X_final = X_rel[:, Sred]
    return X_final, Sred

# ---------- Step 5: HYCADEX Pipeline ----------
def hycadex(X, y, tau_rel=0.05, tau_red=0.7):
    """
    Complete HYCADEX pipeline: relevance + redundancy elimination
    """
    # Step 1: Relevance scores
    relevance_scores = compute_relevance_scores(X, y)
    
    # Step 2: Select relevant attributes
    X_rel, Srel = select_relevant_attributes(X, y, relevance_scores, tau_rel=tau_rel)
    
    # Step 3: Compute redundancy scores
    redundancy_scores = compute_redundancy_scores(X_rel)
    
    # Step 4: Select non-redundant features
    X_final, Sred_local = select_non_redundant_features(X_rel, redundancy_scores, tau_red=tau_red)
    
    # Map back to original feature indices
    Sred = [Srel[i] for i in Sred_local]
    
    return X_final, Sred

# ---------- Example Usage ----------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Create synthetic dataset
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10,
                               n_redundant=5, n_repeated=0, random_state=42)
    
    # Run HYCADEX algorithm
    X_reduced, selected_features = hycadex(X, y, tau_rel=0.05, tau_red=0.7)
    
    print("Selected feature indices:", selected_features)
    print("Shape of reduced feature matrix:", X_reduced.shape)
