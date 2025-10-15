Unified Pipeline for EDM Student Performance Prediction
Algorithms Included:
1. MATRICLUST-FS (Feature Selection)
2. FINSCO (Hybrid Feature Selection with CSO)
3. HYCADEX (Sample Selection)
4. COGNIFUZZ (Fuzzy Neuro PRISM Classifier)
5. OPTIFUZZTREE (Fuzzy Neuro J48 Classifier)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans

# ----------------------------
# 1. MATRICLUST-FS Algorithm
# ----------------------------
def matriclust_fs(X, y, k_clusters=3, su_threshold=0.05):
    """Matrix-driven Clustering Feature Selection"""
    # Compute Symmetric Uncertainty (SU)
    mi = mutual_info_classif(X, y)
    H_X = -np.sum((X/np.sum(X, axis=0)) * np.log2((X+1e-6)/np.sum(X, axis=0)), axis=0)
    H_y = -np.sum(np.bincount(y)/len(y) * np.log2((np.bincount(y)/len(y))+1e-6))
    su_scores = (2 * mi) / (H_X + H_y + 1e-6)
    
    # Remove irrelevant attributes
    relevant_idx = np.where(su_scores >= su_threshold)[0]
    X_relevant = X[:, relevant_idx]
    
    # Non-negative Matrix Factorization (simplified using KMeans for demo)
    kmeans = KMeans(n_clusters=min(k_clusters, X_relevant.shape[1]), random_state=42).fit(X_relevant.T)
    
    # Select representative attribute for each cluster
    X_final_idx = []
    for i in range(kmeans.n_clusters):
        cluster_idx = np.where(kmeans.labels_ == i)[0]
        # Pick feature with highest SU in cluster
        best_feature = cluster_idx[np.argmax(su_scores[relevant_idx][cluster_idx])]
        X_final_idx.append(relevant_idx[best_feature])
    
    X_final = X[:, X_final_idx]
    return X_final, X_final_idx

# ----------------------------
# 2. HYCADEX Algorithm
# ----------------------------
def hycadex(X, y, tau_rel=0.05, tau_red=0.7):
    """Hybrid Sample Selection"""
    # Relevance score
    relevance_scores = mutual_info_classif(X, y)
    rel_idx = np.where(relevance_scores >= tau_rel)[0]
    X_rel = X[:, rel_idx]
    
    # Redundancy: max correlation
    corr_matrix = np.corrcoef(X_rel.T)
    redundancy_scores = np.max(np.abs(corr_matrix - np.eye(len(rel_idx))), axis=1)
    non_redundant_idx_local = np.where(redundancy_scores <= tau_red)[0]
    X_final = X_rel[:, non_redundant_idx_local]
    
    # Map back to original feature indices
    final_idx = [rel_idx[i] for i in non_redundant_idx_local]
    return X_final, final_idx

# ----------------------------
# 3. Fuzzification
# ----------------------------
def fuzzify_features(X, num_membership=3):
    """Fuzzify input features using Gaussian membership functions"""
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

# ----------------------------
# 4. COGNIFUZZ Classifier
# ----------------------------
def generate_fuzzy_rules(X_membership, y):
    """Generate fuzzy rules from fuzzified training data"""
    n_samples = X_membership[0].shape[0]
    n_features = len(X_membership)
    rules = []
    for i in range(n_samples):
        antecedent = [np.argmax(X_membership[j][i]) for j in range(n_features)]
        consequent = y[i]
        rules.append((antecedent, consequent))
    return rules

def classify_cognifuzz(X_test, X_membership_train, rules, num_membership=3):
    """Classify new instances using COGNIFUZZ"""
    n_samples, n_features = X_test.shape
    centers = np.linspace(0, 1, num_membership)
    sigma = 0.1
    membership_matrices_test = []
    
    # Fuzzify test data
    for j in range(n_features):
        feature_values = MinMaxScaler().fit(X_test[:, j].reshape(-1,1)).transform(X_test[:, j].reshape(-1,1)).flatten()
        memberships = np.zeros((n_samples, num_membership))
        for k, c in enumerate(centers):
            memberships[:, k] = np.exp(-0.5 * ((feature_values - c)/sigma)**2)
        membership_matrices_test.append(memberships)
    
    predictions = []
    classes = list(set([rule[1] for rule in rules]))
    
    for i in range(n_samples):
        firing_strengths = []
        for rule in rules:
            antecedent, consequent = rule
            strength = np.prod([membership_matrices_test[j][i, antecedent[j]] for j in range(n_features)])
            firing_strengths.append((strength, consequent))
        
        total_strength = sum([f[0] for f in firing_strengths]) + 1e-6
        normalized = [(f[0]/total_strength, f[1]) for f in firing_strengths]
        
        class_membership = {c:0 for c in classes}
        for s, c in normalized:
            class_membership[c] += s
        
        predicted_class = max(class_membership, key=class_membership.get)
        predictions.append(predicted_class)
    
    return np.array(predictions)

# ----------------------------
# 5. OPTIFUZZTREE Classifier
# ----------------------------
def train_optifuzztree(X_fuzzy, y):
    """Train Fuzzy Neuro J48 Classifier"""
    X_flattened = np.hstack(X_fuzzy)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_flattened, y)
    return clf

def classify_optifuzztree(X_test, X_fuzzy_train, clf, num_membership=3):
    """Classify using OPTIFUZZTREE"""
    n_samples, n_features = X_test.shape
    membership_matrices = []
    centers = np.linspace(0,1,num_membership)
    sigma = 0.1
    for j in range(n_features):
        feature_values = MinMaxScaler().fit(X_test[:, j].reshape(-1,1)).transform(X_test[:, j].reshape(-1,1)).flatten()
        memberships = np.zeros((n_samples, num_membership))
        for k, c in enumerate(centers):
            memberships[:, k] = np.exp(-0.5 * ((feature_values - c)/sigma)**2)
        membership_matrices.append(memberships)
    
    X_test_flattened = np.hstack(membership_matrices)
    predictions = clf.predict(X_test_flattened)
    
    # Compute fuzzy membership approximation using leaf node distribution
    leaf_indices = clf.apply(X_test_flattened)
    tree = clf.tree_
    classes = clf.classes_
    membership_values = np.zeros((n_samples, len(classes)))
    for i, leaf in enumerate(leaf_indices):
        distribution = tree.value[leaf][0]
        membership_values[i, :] = distribution / np.sum(distribution)
    
    return predictions, membership_values

# ----------------------------
# Example Usage Pipeline
# ----------------------------
if __name__ == "__main__":
    # 1. Generate synthetic dataset
    X, y = make_classification(n_samples=200, n_features=10, n_informative=6,
                               n_redundant=2, n_classes=2, random_state=42)
    
    # 2. Apply MATRICLUST-FS
    X_fs, idx_fs = matriclust_fs(X, y)
    print("Selected features by MATRICLUST-FS:", idx_fs)
    
    # 3. Apply HYCADEX
    X_hycadex, idx_hycadex = hycadex(X_fs, y)
    print("Selected features by HYCADEX:", idx_hycadex)
    
    # 4. Fuzzify features
    X_fuzzy, centers = fuzzify_features(X_hycadex, num_membership=3)
    
    # 5. Train and classify using COGNIFUZZ
    rules = generate_fuzzy_rules(X_fuzzy, y)
    preds_cognifuzz = classify_cognifuzz(X_hycadex, X_fuzzy, rules)
    print("COGNIFUZZ Accuracy:", accuracy_score(y, preds_cognifuzz))
    
    # 6. Train and classify using OPTIFUZZTREE
    clf_optifuzz = train_optifuzztree(X_fuzzy, y)
    preds_optifuzz, memberships = classify_optifuzztree(X_hycadex, X_fuzzy, clf_optifuzz)
    print("OPTIFUZZTREE Accuracy:", accuracy_score(y, preds_optifuzz))
    print("Fuzzy Memberships (first 5 samples):\n", memberships[:5])
