import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # Load full data first
    full_df = pd.read_csv(file_path, header=None)

    # First row = setpoints
    setpoints = full_df.iloc[0]
    
    # Remaining rows = actual data
    df = full_df.iloc[1:].copy()
    df.columns = setpoints.index  # use same indices as dummy column names

    return df

def preprocess(file_path):
    df = load_data(file_path)

    # Convert all values to float
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop timestamp (col 0)
    df.drop(columns=df.columns[0], inplace=True)

    # Input features = col 1 to 41
    X = df.iloc[:, 0:41]

    # Labels (primary outputs) = ~.C.Actual in col 42–71 (every 2nd col starting from 42)
    y = df.iloc[:, 42:72:2]  # Skip setpoints (they are at even indices)

    # Drop constant columns
    nunique = X.nunique()
    X = X.loc[:, nunique != 1]

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Run to test
if __name__ == "__main__":
    X, y, scaler = preprocess("data/manufacturing_data.csv")
    print("✅ Preprocessing done.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
