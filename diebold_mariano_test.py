import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import norm

def dm_test(e1, e2, h=1, power=1):
    d = np.abs(e1) ** power - np.abs(e2) ** power
    mean_d = np.mean(d)
    n = len(d)
    gamma = []
    for lag in range(1, h):
        gamma.append(np.corrcoef(d[:-lag], d[lag:])[0, 1])
    V_d = (np.var(d, ddof=1) + 2 * np.sum(gamma)) / n
    DM_stat = mean_d / np.sqrt(V_d)
    p_value = 2 * norm.cdf(-abs(DM_stat))
    return DM_stat, p_value

def load_predictions(file_path, actual_col, pred_col):
    df = pd.read_csv(file_path)
    actual = df[actual_col].values
    pred = df[pred_col].values
    return actual, pred

def get_model_file_and_cols(model):
    mapping = {
        'lstm': ('model_predictions2-lstm.csv', 'actual', 'predicted'),
        'gru': ('model_predictions-gru.csv', 'actual', 'predicted'),
        'gbt': ('model_predictions_gbt.csv', 'actual', 'prediction'),
        'sarima': ('arima_predictions.csv', 'actual_label', 'predicted_label'),
        'random_forest': ('model_predictions_stacked_rf.csv', 'actual', 'stacked_prediction'),
        'adaboost': ('model_predictions_stacked_adaboost.csv', 'actual', 'stacked_prediction'),
        'stacked_gbt': ('model_predictions_stacked_gbt.csv', 'actual', 'stacked_prediction'),
    }
    return mapping[model]

def run_dm_test(model_i, model_j):
    file_i, actual_col_i, pred_col_i = get_model_file_and_cols(model_i)
    file_j, actual_col_j, pred_col_j = get_model_file_and_cols(model_j)
    actual_i, pred_i = load_predictions(file_i, actual_col_i, pred_col_i)
    actual_j, pred_j = load_predictions(file_j, actual_col_j, pred_col_j)
    # Ensure same length and alignment
    n = min(len(actual_i), len(actual_j))
    actual = actual_i[:n]
    errors_i = (pred_i[:n] != actual)
    errors_j = (pred_j[:n] != actual)
    DM_stat, p_value = dm_test(errors_i, errors_j)
    print(f"Diebold-Mariano test between {model_i} and {model_j}:")
    print(f"DM statistic: {DM_stat:.4f}, p-value: {p_value:.4f}")

if __name__ == "__main__":
    models = ['lstm', 'gru', 'gbt', 'sarima', 'random_forest', 'adaboost', 'stacked_gbt']
    n_models = len(models)
    p_matrix = [['' for _ in range(n_models)] for _ in range(n_models)]
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                p_matrix[i][j] = '-'
            else:
                model_i = models[i]
                model_j = models[j]
                file_i, actual_col_i, pred_col_i = get_model_file_and_cols(model_i)
                file_j, actual_col_j, pred_col_j = get_model_file_and_cols(model_j)
                actual_i, pred_i = load_predictions(file_i, actual_col_i, pred_col_i)
                actual_j, pred_j = load_predictions(file_j, actual_col_j, pred_col_j)
                n = min(len(actual_i), len(actual_j))
                actual = actual_i[:n]
                errors_i = (pred_i[:n] != actual)
                errors_j = (pred_j[:n] != actual)
                _, p_value = dm_test(errors_i, errors_j)
                p_matrix[i][j] = f"{p_value:.4f}"
    # Print header
    print(f"{'':<15}" + ''.join([f"{m:<15}" for m in models]))
    print('-' * (15 + 15 * n_models))
    for i, row in enumerate(p_matrix):
        print(f"{models[i]:<15}" + ''.join([f"{val:<15}" for val in row]))
