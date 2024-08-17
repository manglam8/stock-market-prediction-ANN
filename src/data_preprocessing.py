import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """
    Load the CSV file from the specified file path.
    
    Args:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data by selecting relevant columns, handling missing values,
    scaling features, and preparing input-output pairs.
    
    Args:
    df (pd.DataFrame): Raw data.
    
    Returns:
    tuple: Tuple containing the input (X) and output (y) datasets.
    """
    # Select relevant columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    
    # Convert Date to datetime format and sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Scale the features
    scaler = MinMaxScaler()
    df[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']])

    # Prepare input (X) and output (y)
    X = []
    y = []

    for i in range(len(df) - 1):
        X.append(df[['Open', 'High', 'Low', 'Close']].iloc[i].values)
        y.append(df['Close'].iloc[i + 1])

    # Convert to numpy arrays for model training
    X = pd.DataFrame(X, columns=['Open', 'High', 'Low', 'Close']).to_numpy()
    y = pd.Series(y, name='Next_Close').to_numpy()

    return X, y
"""
if __name__ == "__main__":
    # File path to the CSV file in the data folder
    file_path = '~/stock-market-prediction-ANN/data/NIFTY50_Historical_PR_15082014to15082024.csv'

    # Load and preprocess the data
    df = load_data(file_path)
    X, y = preprocess_data(df)

    # Print shapes to verify
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
"""

