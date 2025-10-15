# MATRCLUST-FS: Matrix-driven Clustering for Feature Selection
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

# ---------- Step 1: Symmetric Uncertainty (SU) Calculation ----------
def symmetric_uncertainty(X, y):
    """
    Calculate Symmetric Uncertainty (SU) for each feature in X with respect to target y.
    SU = 2 * MI(Xi, y) / (H(Xi) + H(y))
    """
    # Mutual Information
    mi = mutual_info_classif(X, y, discrete_features='auto')
    
    # Entropy of each attribute
    def entropy(column):
        probs = np.bincount(column) / len(column)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    H_y = entropy(y)
    H_X = np.array([entropy(X[:, i].astype(int)) for i in range(X.shape[1])])
    
    SU = (2 * mi) / (H_X + H_y)
    return SU

# ---------- Step 2: Remove Irrelevant Attributes ----------
def remove_irrelevant_features(X, y, threshold=0.05):
    SU = symmetric_uncertainty(X, y)
    relevant_idx = np.where(SU >= threshold)[0]
    X_relevant = X[:, relevant_idx]
    return X_relevant, relevant_idx, SU[relevant_idx]

# ---------- Step 3: Non-negative Matrix Factorization ----------
def apply_nmf(X, n_components=5, max_iter=200):
    nmf_model = NMF(n_components=n_components, init='random', max_iter=max_iter, random_state=42)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_
    return W, H

# ---------- Step 4: Cluster Attributes ----------
def cluster_attributes(H, n_clusters=5):
    # Cluster rows of H (transposed) using KMeans
    H_T = H.T
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(H_T)
    return cluster_labels

# ---------- Step 5: Remove Redundant Attributes ----------
def select_representative_features(X_relevant, SU_relevant, cluster_labels):
    final_features_idx = []
    for cluster in np.unique(cluster_labels):
        cluster_idx = np.where(cluster_labels == cluster)[0]
        # Pick the feature with the highest SU in the cluster
        best_feature_idx = cluster_idx[np.argmax(SU_relevant[cluster_idx])]
        final_features_idx.append(best_feature_idx)
    X_final = X_relevant[:, final_features_idx]
    return X_final, final_features_idx

# ---------- Complete MATRCLUST-FS Pipeline ----------
def matriclust_fs(X, y, su_threshold=0.05, nmf_components=5, n_clusters=5):
    # Scale X to [0,1] for NMF
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 2: Remove irrelevant attributes
    X_relevant, relevant_idx, SU_relevant = remove_irrelevant_features(X_scaled, y, threshold=su_threshold)
    
    # Step 3: Apply NMF
    W, H = apply_nmf(X_relevant, n_components=nmf_components)
    
    # Step 4: Cluster attributes
    cluster_labels = cluster_attributes(H, n_clusters=n_clusters)
    
    # Step 5: Remove redundant attributes
    X_final, final_idx = select_representative_features(X_relevant, SU_relevant, cluster_labels)
    
    # Map back to original feature indices
    original_feature_idx = relevant_idx[final_idx]
    
    return X_final, original_feature_idx

# ---------- Example Usage ----------
if __name__ == "__main__":
    # Example synthetic dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    
    # Run MATRCLUST-FS
    X_final, selected_features = matriclust_fs(X, y, su_threshold=0.05, nmf_components=5, n_clusters=5)
    
    print("Selected feature indices:", selected_features)
    print("Shape of reduced feature matrix:", X_final.shape)
