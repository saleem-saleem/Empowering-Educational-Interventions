import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

# ---------- Step 1: Fitness Function ----------
def evaluate_fitness(X_subset, y, classifier='J48', cv=5):
    """
    Evaluate feature subset using classification accuracy.
    """
    if classifier == 'Prism':
        # Placeholder: Prism classifier implementation
        clf = DecisionTreeClassifier()  # Using DecisionTree as proxy
    else:
        clf = DecisionTreeClassifier()
        
    scores = cross_val_score(clf, X_subset, y, cv=cv, scoring='accuracy')
    return scores.mean()

# ---------- Step 2: Random Selection ----------
def random_subset(X, p):
    """
    Select p random features from X.
    """
    n_features = X.shape[1]
    selected_idx = np.random.choice(n_features, size=p, replace=False)
    return selected_idx, X[:, selected_idx]

# ---------- Step 3: Cuckoo Search Optimization (CSO) ----------
def cuckoo_search(X_candidates, y, classifier='J48'):
    """
    Select the best attribute among candidates using fitness evaluation.
    """
    best_score = -1
    best_attr = None
    for i in range(X_candidates.shape[1]):
        score = evaluate_fitness(X_candidates[:, i].reshape(-1, 1), y, classifier)
        if score > best_score:
            best_score = score
            best_attr = i
    return best_attr, best_score

# ---------- Step 4: Iterative Feature Subset Optimization ----------
def finsco_irrelevant_features(X, y, T=10, p_fraction=0.5, classifier='J48'):
    """
    Step 1: Remove irrelevant features using random selection + CSO
    """
    n_features = X.shape[1]
    selected_features = []
    
    # Initialize random subset
    initial_idx, X_subset = random_subset(X, max(1, int(n_features * p_fraction)))
    selected_features.extend(initial_idx)
    
    for t in range(T):
        # Random selection from current subset
        p = max(1, int(len(selected_features) * p_fraction))
        random_idx = np.random.choice(selected_features, size=p, replace=False)
        X_candidates = X[:, random_idx]
        
        # CSO selects best attribute
        best_attr_idx, _ = cuckoo_search(X_candidates, y, classifier)
        selected_features = list(set(selected_features + [random_idx[best_attr_idx]]))
        
    return X[:, selected_features], selected_features

# ---------- Step 5: Remove Redundant Features using NMF ----------
def remove_redundant_features(X_selected, n_clusters=5):
    """
    Step 2: Remove redundant features using NMF + correlation
    """
    # Scale to [0,1] for NMF
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Apply NMF
    nmf_model = NMF(n_components=n_clusters, init='random', random_state=42, max_iter=200)
    W = nmf_model.fit_transform(X_scaled)
    H = nmf_model.components_
    
    # Cluster features based on H.T
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(H.T)
    
    final_features_idx = []
    for cluster in np.unique(cluster_labels):
        cluster_idx = np.where(cluster_labels == cluster)[0]
        # Compute correlation among features in cluster
        corr_matrix = np.corrcoef(X_selected[:, cluster_idx].T)
        # Select the feature with the highest average correlation to others
        avg_corr = corr_matrix.mean(axis=0)
        best_feature_idx = cluster_idx[np.argmax(avg_corr)]
        final_features_idx.append(best_feature_idx)
    
    X_final = X_selected[:, final_features_idx]
    return X_final, final_features_idx

# ---------- Complete FINSCO Pipeline ----------
def finsco_algorithm(X, y, T=10, p_fraction=0.5, classifier='J48', n_clusters=5):
    # Step 1: Eliminate irrelevant features
    X_selected, selected_idx = finsco_irrelevant_features(X, y, T=T, p_fraction=p_fraction, classifier=classifier)
    
    # Step 2: Remove redundant features
    X_final, final_idx = remove_redundant_features(X_selected, n_clusters=n_clusters)
    
    # Map back to original feature indices
    original_feature_idx = [selected_idx[i] for i in final_idx]
    
    return X_final, original_feature_idx

# ---------- Example Usage ----------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Synthetic dataset
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    
    # Run FINSCO
    X_final, selected_features = finsco_algorithm(X, y, T=10, p_fraction=0.5, classifier='J48', n_clusters=5)
    
    print("Selected feature indices:", selected_features)
    print("Shape of reduced feature matrix:", X_final.shape)
