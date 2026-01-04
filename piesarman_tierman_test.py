import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

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

def piesarman_tierman_test(errors_i, errors_j):
    # Wilcoxon signed-rank test for paired samples (non-parametric)
    stat, p_value = wilcoxon(errors_i, errors_j)
    return stat, p_value

if __name__ == "__main__":
    models = ['lstm', 'gru', 'gbt', 'sarima', 'random_forest', 'adaboost', 'stacked_gbt']
    print(f"{'Model':<15}{'Wilcoxon p-value':<20}")
    print('-' * 35)
    for model in models:
        file, actual_col, pred_col = get_model_file_and_cols(model)
        actual, pred = load_predictions(file, actual_col, pred_col)
        n = len(actual)
        errors = (pred[:n] != actual).astype(int)
        # Test if median error is zero
        stat, p_value = wilcoxon(errors - 0)
        print(f"{model:<15}{p_value:<20.4f}")
