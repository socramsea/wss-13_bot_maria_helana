#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

print("📥 Carregando modelo...")
model = load_model('models/maria_helena_lstm_trained.h5')

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/config.pkl', 'rb') as f:
    config = pickle.load(f)

features = config['features']
lookback = config['lookback']

print("✅ Modelo carregado!")
print(f"📊 Features: {features}")
print(f"📊 Lookback: {lookback}")

# Carregar dados
df = pd.read_csv('data/market/ticks_BTC-USDT_15m.csv', header=0)
print(f"✅ {len(df):,} linhas carregadas")

# Preparar últimas N linhas
recent = df[features].fillna(0).values[-lookback:]

# Normalizar
recent_scaled = scaler.transform(recent)

# Reshapear
X = recent_scaled.reshape(1, lookback, len(features))

# Predição
print("\n🔮 Fazendo predição...")
pred = model.predict(X, verbose=0)[0]
pred_class = np.argmax(pred)

signals = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
signal = signals[pred_class]

print(f"\n{'='*60}")
print(f"🎯 PREDIÇÃO ATUAL")
print(f"{'='*60}")
print(f"Sinal: {signal}")
print(f"Confiança: {pred[pred_class]*100:.2f}%")
print(f"\nProbabilidades:")
print(f"  SELL: {pred[0]*100:.1f}%")
print(f"  HOLD: {pred[1]*100:.1f}%")
print(f"  BUY:  {pred[2]*100:.1f}%")
print(f"{'='*60}")
