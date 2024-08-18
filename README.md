# stock-market-prediction-ANN
This project aims to predict the Nifty50 index closing price using historical data and artificial neural networks.

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/manglam8/stock-market-prediction-ANN
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```
   python src/train.py
   ```
4. Evaluate the model:
   ```
   python src/evaluate.py
   ```
5. Run the flask app to predict for custom input data
   ```
   python flask_app.py
   ```
   Visit http://127.0.0.1:5000/ in your browser.

## Steps I followed to build :

1. Gather and prepare data (downloaded data directly from NSE website as csv file and did pre-processing)
2. Define feature and labels (current day's features Open, High, Low, Close Price and label as next day's Closing Price)
3. Split Dataset (80% training & 20% testing)
4. Build ANN model (used TensorFlow)
5. Evaluate model (used Mean Absolute Error to measure accuracy)
6. Deploy model (used Flask for a web interface)
7. Iterate and improve (pending)
