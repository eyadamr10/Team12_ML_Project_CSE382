import pandas as pd
from sklearn.model_selection import train_test_split
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