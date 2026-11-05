import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings 
warnings.filterwarnings('ignore')

print("Loading data...")
booking_data = pd.read_csv('data/H2.csv')

print("\n--- EDA SUMMARY ---")
print(f"Dataset Shape: {booking_data.shape}")
print(f"Missing Values:\n{booking_data.isna().sum()[booking_data.isna().sum() > 0]}")

future_info_columns = ['ReservationStatus', 'ReservationStatusDate']
booking_data = booking_data.drop(columns=future_info_columns)

features = booking_data.drop(columns=['ADR', 'IsCanceled'])
price_target = booking_data['ADR']
cancel_target = booking_data['IsCanceled']

features_train, features_val, price_train, price_val, cancel_train, cancel_val = train_test_split(
    features, price_target, cancel_target, test_size=0.2, random_state=42
)


print("\nBuilding Preprocessing Pipelines...")
num_cols = features.select_dtypes(include=['int64', 'float64']).columns
cat_cols = features.select_dtypes(include=['object']).columns

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

data_cleaner = ColumnTransformer(
    transformers=[
        ('numerical', num_pipeline, num_cols),
        ('categorical', cat_pipeline, cat_cols)
    ])

print("\n--- TASK A: REGRESSION (Baseline) ---")
price_model = Pipeline(steps=[
    ('preprocessor', data_cleaner),
    ('regressor', LinearRegression())
])

price_model.fit(features_train, price_train) 
predicted_prices = price_model.predict(features_val)

print(f"MAE:  {mean_absolute_error(price_val, predicted_prices):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(price_val, predicted_prices)):.2f}")
print(f"R2:   {r2_score(price_val, predicted_prices):.2f}")


print("\n--- TASK B: CLASSIFICATION (Baseline) ---")
cancel_model = Pipeline(steps=[
    ('preprocessor', data_cleaner),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

cancel_model.fit(features_train, cancel_train) 
predicted_cancels = cancel_model.predict(features_val)

print(f"Accuracy:  {accuracy_score(cancel_val, predicted_cancels):.2f}")
print(f"Precision: {precision_score(cancel_val, predicted_cancels):.2f}")
print(f"Recall:    {recall_score(cancel_val, predicted_cancels):.2f}")
print(f"F1-score:  {f1_score(cancel_val, predicted_cancels):.2f}")

print("\n--- TASK C: CLUSTERING (Baseline) ---")
segment_cols = ['LeadTime', 'StaysInWeekendNights', 'StaysInWeekNights', 'Adults', 'PreviousCancellations']
segmentation_data = booking_data[segment_cols].dropna()

segmentation_sample = segmentation_data.sample(n=10000, random_state=42)

segmentation_model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=3, random_state=42, n_init=10))
])

cluster_labels = segmentation_model.fit_predict(segmentation_sample)
scaled_sample = segmentation_model.named_steps['scaler'].transform(segmentation_sample)
sil_score = silhouette_score(scaled_sample, cluster_labels)

print(f"K-Means (k=3) Silhouette Score: {sil_score:.2f}")
print("\nMilestone 1 execution complete!")