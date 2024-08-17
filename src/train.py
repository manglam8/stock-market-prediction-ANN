from tensorflow.keras.models import save_model
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, preprocess_data
from src.model import build_model

def train_model(X_train, y_train):
    """
    Train the neural network model.
    
    Args:
    X_train (np.array): Training input data.
    y_train (np.array): Training output data.
    
    Returns:
    history: Training history object.
    """
    model = build_model(X_train.shape[1])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Save the model
    model.save('models/stock_price_model.h5')
    
    return history

if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'data/NIFTY50_Historical_PR_15082014to15082024.csv'
    df = load_data(file_path)
    X, y = preprocess_data(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    history = train_model(X_train, y_train)
    
    # Print training history
    print("Training completed. Final training history:")
    print(history.history)
