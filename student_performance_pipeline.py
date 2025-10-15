student_performance_pipeline.py
Full pipeline for student academic performance prediction using:
- MATRICLUST-FS
- FINSCO
- HYCADEX
- COGNIFUZZ (Fuzzy Neuro Prism)
- OPTIFUZZTREE (Fuzzy Neuro J48)

Includes:
- Feature selection
- Sample selection
- Fuzzy-neuro classification
- Performance evaluation
- Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load Dataset (EPM Dataset)
# -----------------------------
def load_epm_dataset(filepath):
    """
    Load EPM dataset CSV.
    Returns: X (features), y (labels)
    """
    df = pd.read_csv(filepath)
    # Assume last column is the label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

# -----------------------------
# 2. Feature Selection Methods
# -----------------------------
# Placeholder functions: Replace with your actual implementations
def matriclust_fs(X, y):
    """MATRICLUST-FS feature selection"""
    # Select subset of features (example: first half)
    selected_features = X[:, :X.shape[1]//2]
    return selected_features

def finsco(X, y):
    """FINSCO hybrid feature selection"""
    selected_features = X[:, :int(X.shape[1]*0.6)]
    return selected_features

# -----------------------------
# 3. Sample Selection: HYCADEX
# -----------------------------
def hycadex(X, y):
    """HYCADEX sample selection (adaptive sampling)"""
    # Example: select 80% of samples randomly
    idx = np.random.choice(range(X.shape[0]), int(X.shape[0]*0.8), replace=False)
    return X[idx], y[idx]

# -----------------------------
# 4. Fuzzy-Neuro Classifiers
# -----------------------------
# Placeholder functions: Replace with actual COGNIFUZZ and OPTIFUZZTREE code
def cognifuzz_predict(X_train, y_train, X_test):
    """Fuzzy Neuro Prism prediction"""
    # Dummy: Random predictions for demonstration
    return np.random.choice(np.unique(y_train), size=X_test.shape[0])

def optifuzztree_predict(X_train, y_train, X_test):
    """Fuzzy Neuro J48 prediction"""
    return np.random.choice(np.unique(y_train), size=X_test.shape[0])

# -----------------------------
# 5. Performance Metrics
# -----------------------------
def evaluate_model(y_true, y_pred):
    """Compute precision, recall, f1, accuracy"""
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

# -----------------------------
# 6. Pipeline Execution
# -----------------------------
def run_pipeline(X, y, test_size=0.2, random_state=42):
    """
    Run full pipeline:
    1. Feature Selection (MATRICLUST-FS, FINSCO)
    2. Sample Selection (HYCADEX)
    3. Fuzzy-Neuro Classification (COGNIFUZZ, OPTIFUZZTREE)
    4. Performance Evaluation
    """
    results = []

    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Phase 1: MATRICLUST-FS
    X_train_fs = matriclust_fs(X_train_full, y_train_full)
    X_test_fs = X_test[:, :X_train_fs.shape[1]]  # Align features
    y_pred_prism_fs = cognifuzz_predict(X_train_fs, y_train_full, X_test_fs)
    y_pred_j48_fs = optifuzztree_predict(X_train_fs, y_train_full, X_test_fs)
    results.append(('MATRICLUST-FS Prism', evaluate_model(y_test, y_pred_prism_fs)))
    results.append(('MATRICLUST-FS J48', evaluate_model(y_test, y_pred_j48_fs)))

    # Phase 2: FINSCO
    X_train_finsco = finsco(X_train_full, y_train_full)
    X_test_finsco = X_test[:, :X_train_finsco.shape[1]]
    y_pred_prism_finsco = cognifuzz_predict(X_train_finsco, y_train_full, X_test_finsco)
    y_pred_j48_finsco = optifuzztree_predict(X_train_finsco, y_train_full, X_test_finsco)
    results.append(('FINSCO Prism', evaluate_model(y_test, y_pred_prism_finsco)))
    results.append(('FINSCO J48', evaluate_model(y_test, y_pred_j48_finsco)))

    # Phase 3: HYCADEX sample selection
    X_train_hycadex, y_train_hycadex = hycadex(X_train_full, y_train_full)
    X_test_hycadex = X_test[:, :X_train_hycadex.shape[1]] if X_train_hycadex.shape[1] <= X_test.shape[1] else X_test
    y_pred_prism_hycadex = cognifuzz_predict(X_train_hycadex, y_train_hycadex, X_test_hycadex)
    y_pred_j48_hycadex = optifuzztree_predict(X_train_hycadex, y_train_hycadex, X_test_hycadex)
    results.append(('HYCADEX Prism', evaluate_model(y_test, y_pred_prism_hycadex)))
    results.append(('HYCADEX J48', evaluate_model(y_test, y_pred_j48_hycadex)))

    return results, y_test, {
        'MATRICLUST-FS Prism': y_pred_prism_fs,
        'MATRICLUST-FS J48': y_pred_j48_fs,
        'FINSCO Prism': y_pred_prism_finsco,
        'FINSCO J48': y_pred_j48_finsco,
        'HYCADEX Prism': y_pred_prism_hycadex,
        'HYCADEX J48': y_pred_j48_hycadex
    }

# -----------------------------
# 7. Visualization
# -----------------------------
def plot_results(results):
    """Plot metrics for all models"""
    metrics = ['Precision', 'Recall', 'F1-score', 'Accuracy']
    df = pd.DataFrame(results, columns=['Method', 'Metrics'])
    for i, metric in enumerate(metrics):
        values = [res[1][i] for res in results]
        methods = [res[0] for res in results]
        plt.figure(figsize=(10,5))
        sns.barplot(x=methods, y=values)
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        plt.title(f'{metric} Comparison')
        plt.ylim(0,1)
        plt.tight_layout()
        plt.show()

def plot_confusion_matrices(y_true, y_preds):
    """Plot confusion matrices for all models"""
    for method, y_pred in y_preds.items():
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {method}')
        plt.show()

# -----------------------------
# 8. Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load dataset (update path to your CSV file)
    X, y = load_epm_dataset('epm_dataset.csv')

    # Run pipeline
    results, y_true, y_preds = run_pipeline(X, y)

    # Display results
    for method, metrics in results:
        print(f'{method}: Precision={metrics[0]:.3f}, Recall={metrics[1]:.3f}, F1={metrics[2]:.3f}, Accuracy={metrics[3]:.3f}')

    # Plot metrics
    plot_results(results)

    # Plot confusion matrices
    plot_confusion_matrices(y_true, y_preds)
