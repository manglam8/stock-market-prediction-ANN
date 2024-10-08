from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from src.data_preprocessing import load_data, preprocess_data

def evaluate_model(X_test, y_test, scaler):
    """
    Evaluate the trained model.

    Args:
    X_test (np.array): Test input data.
    y_test (np.array): Test output data.
    scaler (MinMaxScaler): Scaler used to inverse transform the predictions and test data.

    Returns:
    None
    """
    # Load the saved model
    model = load_model('models/stock_price_model.h5')

    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform the scaled data to get original values
    y_test_original = scaler.inverse_transform([[0, 0, 0, 0, y] for y in y_test])[:, -1]
    predictions_original = scaler.inverse_transform([[0, 0, 0, 0, p[0]] for p in predictions])[:, -1]

    # Compare predictions with actual values
    mae = mean_absolute_error(y_test_original, predictions_original)
    print(f"Mean Absolute Error on Test Set: {mae}")

    # Print a few sample predictions vs actual values
    print("Predicted vs Actual:")
    for i in range(5):
        print(f"Predicted: {predictions_original[i]}, Actual: {y_test_original[i]}")

if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'data/NIFTY50_Historical_PR_15082014to15082024.csv'
    df = load_data(file_path)
    X, y, scaler = preprocess_data(df)

    # Split the data into training and testing sets
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate the model
    evaluate_model(X_test, y_test, scaler)
