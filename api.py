from flask import Flask, jsonify
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Carregar modelo uma vez
model = load_model('models/maria_helena_lstm_trained.keras')
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/config.pkl', 'rb') as f:
    config = pickle.load(f)

features = config['features']
lookback = config['lookback']

@app.route('/predict', methods=['GET'])
def predict():
    df = pd.read_csv('data/market/ticks_BTC-USDT_15m.csv', header=0)
    recent = df[features].fillna(0).values[-lookback:]
    recent_scaled = scaler.transform(recent)
    X = recent_scaled.reshape(1, lookback, len(features))
    
    pred = model.predict(X, verbose=0)[0]
    signal = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[np.argmax(pred)]
    
    return jsonify({
        'signal': signal,
        'confidence': float(pred.max()),
        'probabilities': {
            'SELL': float(pred[0]),
            'HOLD': float(pred[1]),
            'BUY': float(pred[2])
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
