import pandas as pd
import numpy as np
from scipy.stats import norm

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

def pesaran_timmermann_test(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)
    T = len(actual)
    # Observed accuracy
    I = (actual == pred).astype(int)
    p_hat = I.mean()
    # Marginal probabilities
    p_y1 = np.mean(actual == 1)
    p_y0 = 1 - p_y1
    p_f1 = np.mean(pred == 1)
    p_f0 = 1 - p_f1
    # Expected accuracy under independence
    p_star = p_f1 * p_y1 + p_f0 * p_y0
    # Variance estimate (Pesaran–Timmermann)
    var_p = (p_star * (1 - p_star)) / T
    if var_p == 0:
        pt_stat = 0
        p_value = 1.0
    else:
        pt_stat = (p_hat - p_star) / np.sqrt(var_p)
        p_value = 1 - norm.cdf(pt_stat)  # one-sided
    return pt_stat, p_value

if __name__ == "__main__":
    models = ['lstm', 'gru', 'gbt', 'sarima', 'random_forest', 'adaboost', 'stacked_gbt']
    print(f"{'Model':<15}{'PT p-value':<20}")
    print('-' * 35)
    for model in models:
        file, actual_col, pred_col = get_model_file_and_cols(model)
        actual, pred = load_predictions(file, actual_col, pred_col)
        n = len(actual)
        PT_stat, p_value = pesaran_timmermann_test(actual[:n], pred[:n])
        print(f"{model:<15}{p_value:<20.4f}")
