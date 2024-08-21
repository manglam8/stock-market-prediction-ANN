from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import csv

# Use non-GUI backend for Matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load your trained model
model = load_model('models/stock_price_model.h5')

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def create_plot(dates, actual_closes, prediction, date_processed):
    """Function to create and save the plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual_closes, label='Actual Close Prices')
    plt.scatter(date_processed, prediction, color='red', label='Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    
    # Save the plot as an image
    plot_path = 'static/prediction_plot.png'
    plt.savefig(plot_path)
    plt.close()  # Close the plot to avoid memory issues
    return plot_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    date = request.form['date']
    open_price = float(request.form['open_price'])
    close_price = float(request.form['close_price'])
    high_price = float(request.form['high_price'])
    low_price = float(request.form['low_price'])

    # Process the date
    date_processed = (pd.to_datetime(date) - pd.to_datetime('2014-08-15')).days

    # Prepare the input data for the model
    input_data = np.array([[date_processed, open_price, close_price, high_price, low_price]])
    scaled_input = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(scaled_input)[0][0]

    # Rescale prediction back to the original price range
    predicted_price = scaler.inverse_transform([[0, 0, 0, 0, prediction]])[0][-1]

    # Dummy data for dates and actual closes for visualization
    dates = list(range(date_processed - 10, date_processed))  # Adjust as needed
    actual_closes = [close_price - i for i in range(10)]  # Adjust as needed

    # Create and save the plot
    plot_url = create_plot(dates, actual_closes, predicted_price, date_processed)

    return render_template('index.html', prediction=predicted_price, plot_url=plot_url)

@app.route('/feedback', methods=['POST'])
def feedback():
    # Get the data from the feedback form
    date = request.form['next_date']
    open_price = float(request.form['next_open_price'])
    high_price = float(request.form['next_high_price'])
    low_price = float(request.form['next_low_price'])
    actual_close_price = float(request.form['actual_close_price'])


    # Create or append to feedback_data.csv
    feedback_file = 'data/feedback_data.csv'
    file_exists = os.path.isfile(feedback_file)

    # Append the data to the CSV
    with open(feedback_file, 'a', newline='') as csvfile:
        fieldnames = ['Date', 'Open', 'High', 'Low', 'Close']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Write the header only if the file is new
        
        writer.writerow({
            'Date': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': actual_close_price
        })

    return "Thank you for your feedback!"

if __name__ == "__main__":
    # Ensure the 'static' folder exists for plot saving
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('data'):
        os.makedirs('data')
    
    app.run(debug=True)
