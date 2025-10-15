Analyze and visualize performance metrics for student performance prediction.
Includes Phase 1-4 results for different algorithms and classifiers.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------
# 1. Phase 1: MATRICLUST-FS
# --------------------------
phase1 = {
    'Classifier': ['Prism', 'J48'],
    'Without_FS_TPR': [0.92, 0.89],
    'Without_FS_TNR': [0.21, 0.45],
    'Without_FS_Accuracy': [0.89, 0.88],
    'MATRICLUST_FS_TPR': [0.99, 0.97],
    'MATRICLUST_FS_TNR': [0.53, 0.75],
    'MATRICLUST_FS_Accuracy': [0.98, 0.97]
}

# --------------------------
# 2. Phase 2: FINSCO
# --------------------------
phase2 = {
    'Classifier': ['Prism', 'J48'],
    'Without_FS': [65, 86],
    'MATRICLUST_FS': [70, 89],
    'FINSCO': [79, 90]
}

# --------------------------
# 3. Phase 3: HYCADEX
# --------------------------
phase3 = {
    'Classifier': ['Prism', 'J48'],
    'BHFS': [75, 90],
    'CPESS': [79, 95],
    'KD_MPAS': [85, 96]
}

# --------------------------
# 4. Phase 4: Fuzzy-Neuro Enhanced
# --------------------------
phase4 = {
    'Classifier': ['Prism', 'J48'],
    'Baseline_TPR': [0.85, 0.95],
    'Baseline_TNR': [0.82, 0.95],
    'Baseline_Accuracy': [0.88, 0.95],
    'COGNIFUZZ_TPR': [0.88, None],  # Only applied to Prism
    'COGNIFUZZ_TNR': [0.88, None],
    'COGNIFUZZ_Accuracy': [0.90, None],
    'OPTIFUZZTREE_TPR': [0.99, 0.99],
    'OPTIFUZZTREE_TNR': [0.97, 0.97],
    'OPTIFUZZTREE_Accuracy': [0.97, 0.97]
}

# --------------------------
# Function to plot bar charts
# --------------------------
def plot_phase_results(phase_dict, phase_name, metrics=None):
    """
    Plot bar charts for a given phase.
    
    phase_dict : dict
        Dictionary with classifiers and metric values
    phase_name : str
        Name of the phase for plot title
    metrics : list of str
        List of metrics to plot
    """
    df = pd.DataFrame(phase_dict)
    classifiers = df['Classifier'].values
    x = np.arange(len(classifiers))  # label locations
    width = 0.2
    
    if not metrics:
        metrics = [col for col in df.columns if col != 'Classifier']

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, df[metric].values, width, label=metric)
    
    ax.set_xticks(x + width*(len(metrics)-1)/2)
    ax.set_xticklabels(classifiers)
    ax.set_ylim(0, 1.1 if max(df[metrics].max()) <= 1 else 100)
    ax.set_ylabel('Performance' if max(df[metrics].max()) <= 1 else 'Accuracy (%)')
    ax.set_title(f'{phase_name} Performance Comparison')
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --------------------------
# Plot Phase 1
# --------------------------
plot_phase_results(phase1, 'Phase 1: MATRICLUST-FS', metrics=[
    'Without_FS_TPR', 'Without_FS_TNR', 'Without_FS_Accuracy',
    'MATRICLUST_FS_TPR', 'MATRICLUST_FS_TNR', 'MATRICLUST_FS_Accuracy'
])

# --------------------------
# Plot Phase 2
# --------------------------
plot_phase_results(phase2, 'Phase 2: FINSCO', metrics=['Without_FS', 'MATRICLUST_FS', 'FINSCO'])

# --------------------------
# Plot Phase 3
# --------------------------
plot_phase_results(phase3, 'Phase 3: HYCADEX', metrics=['BHFS', 'CPESS', 'KD_MPAS'])

# --------------------------
# Plot Phase 4
# --------------------------
plot_phase_results(phase4, 'Phase 4: Fuzzy-Neuro Enhanced', metrics=[
    'Baseline_TPR', 'Baseline_TNR', 'Baseline_Accuracy',
    'COGNIFUZZ_TPR', 'COGNIFUZZ_TNR', 'COGNIFUZZ_Accuracy',
    'OPTIFUZZTREE_TPR', 'OPTIFUZZTREE_TNR', 'OPTIFUZZTREE_Accuracy'
])
