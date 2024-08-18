from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_shape):
    """
    Build and compile a neural network model.

    Args:
    input_shape (int): Number of features in the input data (e.g., 4 for Open, High, Low, Close).

    Returns:
    model: Compiled Keras model.
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    
    # Hidden layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    
    # Output layer
    model.add(Dense(1))  # Output layer with 1 unit (predicting next day's closing price)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model

