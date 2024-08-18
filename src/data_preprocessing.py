import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
    # df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert date to number of days since a reference date (e.g., '2014-08-15')
    df['Date'] = (pd.to_datetime(df['Date']) - pd.to_datetime('2014-08-15')).dt.days

    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Date', 'Open', 'High', 'Low', 'Close']])

    # Prepare input (X) and output (y)
    X = []
    y = []

    for i in range(len(scaled_data) - 1):
        X.append(scaled_data[i])
        y.append(scaled_data[i + 1][-1])  # Next day's closing price

    return np.array(X), np.array(y), scaler
