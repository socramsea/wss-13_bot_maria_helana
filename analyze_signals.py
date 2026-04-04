#!/usr/bin/env python3
"""
Maria Helena - Analisar e Gerar Sinais Reais
"""

import pandas as pd
import numpy as np

print("="*80)
print("🔍 MARIA HELENA - ANÁLISE DE SINAIS")
print("="*80)

# Carregar
df = pd.read_csv('data/market/ticks_BTC-USDT_15m.csv', header=0)

print(f"\n📂 Dataset carregado: {len(df):,} linhas")
print(f"📊 Colunas: {list(df.columns)}")

# ============= VERIFICAR action_strategy =============

if 'action_strategy' in df.columns:
    print(f"\n🔍 COLUNA 'action_strategy':")
    print(f"   Valores únicos: {df['action_strategy'].unique()}")
    print(f"   Distribuição:")
    print(df['action_strategy'].value_counts())
    print(f"\n   NaN: {df['action_strategy'].isna().sum()}")
else:
    print(f"\n❌ Coluna 'action_strategy' NÃO EXISTE!")

# ============= VER PRIMEIRAS LINHAS =============

print(f"\n📊 PRIMEIRAS 20 LINHAS:")
print(df[['ts', 'price', 'rsi_norm', 'ma_fast_norm', 'ma_slow_norm', 'action_strategy']].head(20))

print(f"\n📊 ÚLTIMAS 20 LINHAS:")
print(df[['ts', 'price', 'rsi_norm', 'ma_fast_norm', 'ma_slow_norm', 'action_strategy']].tail(20))

# ============= ESTATÍSTICAS =============

print(f"\n📈 ESTATÍSTICAS DOS INDICADORES:")
print(f"   RSI normilizado (0-1):")
print(f"      Min: {df['rsi_norm'].min():.4f}, Max: {df['rsi_norm'].max():.4f}, Média: {df['rsi_norm'].mean():.4f}")
print(f"   MA Fast (EMA9) normalizado:")
print(f"      Min: {df['ma_fast_norm'].min():.4f}, Max: {df['ma_fast_norm'].max():.4f}")
print(f"   MA Slow (EMA21) normalizado:")
print(f"      Min: {df['ma_slow_norm'].min():.4f}, Max: {df['ma_slow_norm'].max():.4f}")

# ============= GERAR SINAIS REAIS =============

print(f"\n🔧 GERANDO SINAIS BASEADOS EM LÓGICA:")

signals = []

for idx, row in df.iterrows():
    score = 0
    
    # RSI estratégia (0-1 normalizado)
    rsi = row['rsi_norm']
    if rsi < 0.35:  # RSI baixo = compra
        score += 2
    elif rsi > 0.65:  # RSI alto = venda
        score -= 2
    
    # EMA estratégia
    ema_fast = row['ma_fast_norm']
    ema_slow = row['ma_slow_norm']
    if ema_fast > ema_slow:  # Uptrend
        score += 1.5
    elif ema_fast < ema_slow:  # Downtrend
        score -= 1.5
    
    # Momentum
    momentum = row['momentum_norm']
    if momentum > 0.6:  # Momentum positivo
        score += 1
    elif momentum < 0.4:  # Momentum negativo
        score -= 1
    
    # Volatilidade
    volatility = row['volatility_norm']
    if volatility > 0.7:  # Alto risco
        score *= 0.8  # Reduz confiança
    
    # Definir sinal
    if score >= 2:
        signal = 'BUY'
    elif score <= -2:
        signal = 'SELL'
    else:
        signal = 'HOLD'
    
    signals.append({'index': idx, 'score': score, 'signal': signal})

# Converter para DataFrame
signals_df = pd.DataFrame(signals)

print(f"\n✅ SINAIS GERADOS:")
print(signals_df['signal'].value_counts())

print(f"\n📊 Distribuição:")
for sig in ['BUY', 'HOLD', 'SELL']:
    count = (signals_df['signal'] == sig).sum()
    pct = count / len(signals_df) * 100
    print(f"   {sig:5s}: {count:5,d} ({pct:5.1f}%)")

# ============= SALVAR =============

df['new_signal'] = signals_df['signal'].values
df['score'] = signals_df['score'].values

# Salvar novo CSV
df.to_csv('data/market/ticks_BTC-USDT_15m_with_signals.csv', index=False)
print(f"\n✅ Novo CSV com sinais salvo: ticks_BTC-USDT_15m_with_signals.csv")

# Verificar
print(f"\n✅ VERIFICAÇÃO FINAL:")
print(f"   Linhas: {len(df):,}")
print(f"   Colunas: {len(df.columns)}")
print(df[['ts', 'price', 'rsi_norm', 'ma_fast_norm', 'ma_slow_norm', 'new_signal']].head(10))

print(f"\n{'='*80}")
print(f"✅ PRONTO! Use 'ticks_BTC-USDT_15m_with_signals.csv' para treinar!")
print(f"{'='*80}")
