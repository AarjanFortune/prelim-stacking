
# import pandas as pd
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np

# # Load predictions from both models
# lstm_df = pd.read_csv('model_predictions2-lstm.csv')
# gru_df = pd.read_csv('model_predictions-gru.csv')

# # Load features from stock_with_sentiment-aggregated.csv
# feat_df = pd.read_csv('stock_with_sentiment-aggregated.csv')

# # Clean and select relevant columns
# feat_df = feat_df[['date', 'c', 'psar', 'PosDI', 'NegDI', 'ADX', 'accumulated_sentiment']]

# # Convert date columns to string for safe merging
# lstm_df['date'] = lstm_df['date'].astype(str)
# gru_df['date'] = gru_df['date'].astype(str)
# feat_df['date'] = feat_df['date'].astype(str)

# # Merge on date to align predictions and features
# merged = pd.merge(lstm_df, gru_df, on='date', suffixes=('_lstm', '_gru'))
# merged = pd.merge(merged, feat_df, on='date', how='left')

# # Features: predicted_prob from both models + c, psar, PosDI, NegDI, ADX, accumulated_sentiment
# feature_cols = [
#     'predicted_prob_lstm',
#     'predicted_prob_gru',
#     'c', 'psar', 'PosDI', 'NegDI', 'ADX', 'accumulated_sentiment'
# ]

# # Convert all features to numeric (in case of missing/empty values)
# for col in feature_cols:
#     merged[col] = pd.to_numeric(merged[col], errors='coerce')

# # Fill missing values (if any) with median
# merged[feature_cols] = merged[feature_cols].fillna(merged[feature_cols].median())

# X = merged[feature_cols].values
# y = merged['actual_lstm'].values  # or 'actual_gru', should be the same

# # Train/test split (use last 20% for test)
# split = int(0.8 * len(X))
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]

# # Stacking with AdaBoost
# clf = AdaBoostClassifier(n_estimators=50, random_state=42)
# clf.fit(X_train, y_train)

# # Predict
# y_pred = clf.predict(X_test)

# # Results
# print('Accuracy:', accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Save predictions
# stacked_pred = pd.DataFrame({
#     'date': merged['date'].iloc[split:],
#     'actual': y_test,
#     'stacked_pred': y_pred
# })
# stacked_pred.to_csv('model_predictions_stacked_adaboost.csv', index=False)
# print('Stacked predictions saved to model_predictions_stacked_adaboost.csv')


# import pandas as pd
# import numpy as np
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import StandardScaler

# # --------------------------------------------------
# # 1. LOAD DATA
# # --------------------------------------------------

# lstm_df = pd.read_csv('model_predictions2-lstm.csv')
# gru_df = pd.read_csv('model_predictions-gru.csv')
# arima_df = pd.read_csv('model_predictions-arima.csv')
# feat_df = pd.read_csv('stock_with_sentiment-aggregated.csv')

# # --------------------------------------------------
# # 2. SELECT FEATURES
# # --------------------------------------------------

# feat_df = feat_df[['date', 'c', 'psar', 'PosDI', 'NegDI', 'ADX', 'accumulated_sentiment']]

# # Convert dates to string for safe merge
# for df in [lstm_df, gru_df, arima_df, feat_df]:
#     df['date'] = df['date'].astype(str)

# # --------------------------------------------------
# # 3. MERGE ALL SOURCES
# # --------------------------------------------------

# merged = pd.merge(lstm_df, gru_df, on='date', suffixes=('_lstm', '_gru'))
# merged = pd.merge(merged, arima_df, on='date', how='left')
# merged = pd.merge(merged, feat_df, on='date', how='left')

# # --------------------------------------------------
# # 4. ARIMA FEATURE ENGINEERING
# # --------------------------------------------------

# # ARIMA direction (already provided)
# merged.rename(columns={'predicted_label': 'arima_direction'}, inplace=True)

# # ARIMA expected return
# merged['arima_return'] = (merged['forecast_price'] - merged['c']) / merged['c']

# # --------------------------------------------------
# # 5. FINAL FEATURE SET
# # --------------------------------------------------

# feature_cols = [
#     'predicted_prob_lstm',
#     'predicted_prob_gru',
#     'arima_direction',
#     'arima_return',
#     'c',
#     'psar',
#     'PosDI',
#     'NegDI',
#     'ADX',
#     'accumulated_sentiment'
# ]

# # Convert to numeric
# for col in feature_cols:
#     merged[col] = pd.to_numeric(merged[col], errors='coerce')

# # Fill missing values with median
# merged[feature_cols] = merged[feature_cols].fillna(
#     merged[feature_cols].median()
# )

# # --------------------------------------------------
# # 6. TRAIN / TEST SPLIT (TIME-AWARE)
# # --------------------------------------------------

# X = merged[feature_cols].values
# y = merged['actual_lstm'].values  # same as actual_gru

# split = int(0.8 * len(X))
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]

# # --------------------------------------------------
# # 7. FEATURE SCALING (IMPORTANT)
# # --------------------------------------------------

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # --------------------------------------------------
# # 8. META-LEARNER (STACKER)
# # --------------------------------------------------

# clf = AdaBoostClassifier(
#     n_estimators=100,
#     learning_rate=0.8,
#     random_state=42
# )

# clf.fit(X_train, y_train)

# # --------------------------------------------------
# # 9. EVALUATION
# # --------------------------------------------------

# y_pred = clf.predict(X_test)

# print('Accuracy:', accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # --------------------------------------------------
# # 10. SAVE STACKED OUTPUT
# # --------------------------------------------------

# stacked_pred = pd.DataFrame({
#     'date': merged['date'].iloc[split:],
#     'actual': y_test,
#     'stacked_pred': y_pred
# })

# stacked_pred.to_csv(
#     'model_predictions_stacked_adaboost_arima.csv',
#     index=False
# )

# print('Saved to model_predictions_stacked_adaboost_arima.csv')


import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

lstm_df = pd.read_csv('model_predictions2-lstm.csv')
gru_df = pd.read_csv('model_predictions-gru.csv')
gbt_df = pd.read_csv('model_predictions_gbt.csv')
arima_df = pd.read_csv('arima_predictions.csv')
feat_df = pd.read_csv('stock_with_sentiment-aggregated.csv')

# --------------------------------------------------
# 2. SELECT TECHNICAL + SENTIMENT FEATURES
# --------------------------------------------------

feat_df = feat_df[['date', 'c', 'psar', 'PosDI', 'NegDI', 'ADX', 'accumulated_sentiment']]

# Convert dates to string for safe merging
for df in [lstm_df, gru_df, gbt_df, arima_df, feat_df]:
    df['date'] = df['date'].astype(str)

# --------------------------------------------------
# 3. MERGE BASE MODEL OUTPUTS
# --------------------------------------------------

merged = pd.merge(lstm_df, gru_df, on='date', suffixes=('_lstm', '_gru'))
merged = pd.merge(merged, gbt_df, on='date', suffixes=('', '_gbt'))
merged = pd.merge(merged, arima_df, on='date', how='left')
merged = pd.merge(merged, feat_df, on='date', how='left')

# --------------------------------------------------
# 4. CLEAN & RENAME COLUMNS
# --------------------------------------------------

# Rename GBT probability
merged.rename(columns={'predicted_prob': 'predicted_prob_gbt'}, inplace=True)

# Rename ARIMA direction
merged.rename(columns={'predicted_label': 'arima_direction'}, inplace=True)

# --------------------------------------------------
# 5. ARIMA FEATURE ENGINEERING
# --------------------------------------------------

merged['arima_return'] = (merged['forecast_price'] - merged['c']) / merged['c']

# --------------------------------------------------
# 6. FINAL FEATURE SET
# --------------------------------------------------

feature_cols = [
    'predicted_prob_lstm',
    'predicted_prob_gru',
    'predicted_prob_gbt',
    'arima_direction',
    'arima_return',
    'c',
    'psar',
    'PosDI',
    'NegDI',
    'ADX',
    'accumulated_sentiment'
]

# Convert everything to numeric
for col in feature_cols:
    merged[col] = pd.to_numeric(merged[col], errors='coerce')

# Fill missing values
merged[feature_cols] = merged[feature_cols].fillna(
    merged[feature_cols].median()
)

# --------------------------------------------------
# 7. TRAIN / TEST SPLIT (TIME-AWARE)
# --------------------------------------------------

X = merged[feature_cols].values
y = merged['actual_lstm'].values  # same label across models

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 8. FEATURE SCALING (CRITICAL)
# --------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# 9. META-LEARNER (STACKER)
# --------------------------------------------------

clf = AdaBoostClassifier(
    n_estimators=150,
    learning_rate=0.7,
    random_state=42
)

clf.fit(X_train, y_train)

# --------------------------------------------------
# 10. EVALUATION
# --------------------------------------------------

y_pred = clf.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 11. SAVE STACKED PREDICTIONS
# --------------------------------------------------

stacked_pred = pd.DataFrame({
    'date': merged['date'].iloc[split:],
    'actual': y_test,
    'stacked_pred': y_pred
})

stacked_pred.to_csv(
    'model_predictions_stacked_adaboost_lstm_gru_gbt_arima.csv',
    index=False
)

print('Saved to model_predictions_stacked_adaboost_lstm_gru_gbt_arima.csv')

