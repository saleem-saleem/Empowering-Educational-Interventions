import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# 1. Instance sizes
# -----------------------------
instances = np.array([1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])

# -----------------------------
# 2. Performance metrics (Tables 15-18)
# -----------------------------
precision = {
    'SPRAR': [0.689,0.681,0.651,0.611,0.696,0.762,0.702,0.775,0.742,0.820],
    'HLVQ': [0.733,0.777,0.740,0.722,0.808,0.862,0.761,0.839,0.839,0.877],
    'MSFMBDNN': [0.869,0.891,0.827,0.756,0.830,0.867,0.757,0.882,0.940,0.933],
    'MATRICLUST_FS': [0.905,0.927,0.902,0.892,0.918,0.880,0.859,0.918,0.954,0.932],
    'FINSCO': [0.937,0.953,0.927,0.911,0.933,0.893,0.882,0.947,0.965,0.961],
    'HYCADEX': [0.948,0.964,0.955,0.954,0.938,0.965,0.972,0.979,0.983,0.985]
}

recall = {
    'SPRAR': [0.821,0.833,0.840,0.846,0.841,0.844,0.856,0.858,0.867,0.869],
    'HLVQ': [0.868,0.875,0.869,0.866,0.867,0.869,0.872,0.875,0.883,0.887],
    'MSFMBDNN': [0.909,0.916,0.912,0.903,0.905,0.914,0.925,0.926,0.929,0.939],
    'MATRICLUST_FS': [0.945,0.956,0.955,0.947,0.954,0.958,0.957,0.966,0.965,0.978],
    'FINSCO': [0.957,0.968,0.967,0.956,0.961,0.966,0.967,0.974,0.977,0.986],
    'HYCADEX': [0.968,0.979,0.975,0.968,0.978,0.974,0.979,0.982,0.985,0.991]
}

f1score = {
    'SPRAR': [0.749,0.748,0.733,0.709,0.761,0.811,0.771,0.814,0.799,0.843],
    'HLVQ': [0.794,0.823,0.799,0.787,0.836,0.865,0.812,0.856,0.869,0.882],
    'MSFMBDNN': [0.888,0.903,0.867,0.822,0.865,0.889,0.832,0.903,0.934,0.934],
    'MATRICLUST_FS': [0.924,0.941,0.927,0.918,0.935,0.917,0.905,0.941,0.959,0.954],
    'FINSCO': [0.946,0.96,0.946,0.932,0.946,0.928,0.922,0.96,0.97,0.973],
    'HYCADEX': [0.957,0.971,0.964,0.965,0.957,0.969,0.975,0.979,0.984,0.987]
}

accuracy = {
    'SPRAR': [0.755,0.757,0.746,0.729,0.769,0.803,0.779,0.817,0.805,0.845],
    'HLVQ': [0.801,0.826,0.804,0.794,0.838,0.866,0.817,0.857,0.861,0.882],
    'MSFMBDNN': [0.889,0.904,0.870,0.829,0.868,0.890,0.841,0.904,0.935,0.936],
    'MATRICLUST_FS': [0.925,0.942,0.928,0.919,0.936,0.919,0.908,0.942,0.960,0.955],
    'FINSCO': [0.947,0.961,0.947,0.934,0.947,0.930,0.925,0.961,0.971,0.974],
    'HYCADEX': [0.958,0.972,0.965,0.961,0.958,0.970,0.976,0.981,0.984,0.986]
}

# -----------------------------
# 3. Plot Performance Metrics
# -----------------------------
def plot_metrics(instances, metrics_dict, title, ylabel):
    plt.figure(figsize=(12,6))
    for method, values in metrics_dict.items():
        plt.plot(instances, values, marker='o', label=method)
    plt.title(title)
    plt.xlabel('Number of Instances')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_metrics(instances, precision, 'Precision Comparison on EPM Dataset', 'Precision')
plot_metrics(instances, recall, 'Recall Comparison on EPM Dataset', 'Recall')
plot_metrics(instances, f1score, 'F1-score Comparison on EPM Dataset', 'F1-score')
plot_metrics(instances, accuracy, 'Accuracy Comparison on EPM Dataset', 'Accuracy')

# -----------------------------
# 4. Error rate
# -----------------------------
def compute_error_rate(acc_dict):
    error_rate = {}
    for method, values in acc_dict.items():
        error_rate[method] = [1 - v for v in values]
    return error_rate

error_rates = compute_error_rate(accuracy)
plot_metrics(instances, error_rates, 'Error Rate Comparison on EPM Dataset', 'Error Rate')

# -----------------------------
# 5. Convergence iterations (Figure 11-12)
# -----------------------------
iterations = {
    'MATRICLUST_FS': [20,21,22,23,24,25,26,27,28,32],
    'FINSCO': [18,19,20,22,24,25,27,29,30,31],
    'HYCADEX': [15,16,17,18,20,21,22,23,25,26]
}

plot_metrics(instances, iterations, 'Average Iterations for Convergence', 'Iterations')

# -----------------------------
# 6. Confusion matrices (example for 10,000 instances)
# -----------------------------
conf_matrices = {
    'MATRICLUST_FS': np.array([[9600, 384],[204, 9600]]),
    'FINSCO': np.array([[9860, 140],[140, 9860]]),
    'HYCADEX': np.array([[9854, 146],[121, 9854]])
}

for method, cm in conf_matrices.items():
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail','Pass'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {method}')
    plt.show()

# -----------------------------
# 7. Computation time (approx, based on description)
# -----------------------------
computation_time = {
    'SPPM': [9.0]*10,
    'SPRAR': [9.3]*10,
    'MSFMBDNN_LSTM': [17.0]*10,
    'MATRICLUST_FS': [6.0,6.1,6.2,6.3,6.5,6.6,6.7,6.8,6.9,7.0],
    'FINSCO': [5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7],
    'HYCADEX': [5.5,5.6,5.7,5.8,6.0,6.2,6.4,6.5,6.6,6.8]
}

plot_metrics(instances, computation_time, 'Computation Time Comparison', 'Time (seconds)')

print("Performance evaluation and visualizations completed successfully!")
