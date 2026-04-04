

#!/usr/bin/env python3
"""
Maria Helena - Preparação de Dados
Renomear, limpar e organizar CSV
"""

import pandas as pd
import numpy as np
import sys

# ============= CONFIGURAÇÕES =============

INPUT_CSV = 'data/market/ticks_BTC-USDT_15m.csv'
OUTPUT_CSV = 'data/market/ticks_BTC-USDT_15m_prepared.csv'

# Mapa de renomeação (seu CSV → formato padrão)
RENAME_MAP = {
    'price_norm': 'price',
    'volume_norm': 'volume',
    'rsi_norm': 'rsi',
    'macd_norm': 'macd',
    'bb_upper_norm': 'bb_upper',
    'bb_middle_norm': 'bb_middle',
    'bb_lower_norm': 'bb_lower',
    # Adicione mais conforme necessário
    'action_strategy': 'signal'  # ← RENAME DA COLUNA DE AÇÃO!
}

# ============= EXECUTAR =============

print("="*80)
print("📊 MARIA HELENA - PREPARAÇÃO DE DADOS")
print("="*80)

# 1. Carregar
print(f"\n📥 Carregando: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, header=0)
print(f"   ✅ {len(df):,} linhas carregadas")
print(f"   ✅ {len(df.columns)} colunas")

# 2. Mostrar antes
print(f"\n📊 ANTES DO RENOMEAR:")
print(f"   Colunas: {list(df.columns)[:10]}")

# 3. Verificar quais colunas existem
print(f"\n🔍 VERIFICANDO MAPA DE RENOMEAÇÃO:")
for old_name, new_name in RENAME_MAP.items():
    if old_name in df.columns:
        print(f"   ✅ {old_name:25s} → {new_name}")
    else:
        print(f"   ❌ {old_name:25s} (NÃO ENCONTRADA!)")

# 4. Renomear só as que existem
rename_dict = {old: new for old, new in RENAME_MAP.items() if old in df.columns}
df.rename(columns=rename_dict, inplace=True)

print(f"\n✅ Colunas renomeadas: {len(rename_dict)}")

# 5. Mostrar depois
print(f"\n📊 DEPOIS DO RENOMEAR:")
print(f"   Colunas: {list(df.columns)[:10]}")

# 6. Limpeza de Dados
print(f"\n🧹 LIMPANDO DADOS:")

# Remover duplicatas
print(f"   • Linhas antes: {len(df):,}")
df = df.drop_duplicates()
print(f"   • Linhas após remover duplicatas: {len(df):,}")

# NaN
nan_count = df.isna().sum().sum()
print(f"   • NaN total: {nan_count}")
if nan_count > 0:
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    print(f"   • NaN após preenchimento: {df.isna().sum().sum()}")

# Infinitos
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"   • Infinitos: {inf_count}")
if inf_count > 0:
    df = df.replace([np.inf, -np.inf], 0)

# 7. Verificar tipo de dados
print(f"\n📊 TIPOS DE DADOS:")
print(df.dtypes)

# 8. Estatísticas
print(f"\n📈 ESTATÍSTICAS:")
print(df.describe())

# 9. Distribuição da coluna signal
if 'signal' in df.columns:
    print(f"\n🎯 DISTRIBUIÇÃO DO SINAL:")
    signal_dist = df['signal'].value_counts()
    for signal, count in signal_dist.items():
        pct = count / len(df) * 100
        print(f"   • {signal:10s}: {count:7,d} ({pct:5.1f}%)")

# 10. Salvar
print(f"\n💾 Salvando: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)
print(f"   ✅ {len(df):,} linhas salvas")

# 11. Confirmar
df_check = pd.read_csv(OUTPUT_CSV)
print(f"\n✅ VERIFICAÇÃO:")
print(f"   • Arquivo criado: {len(df_check):,} linhas")
print(f"   • Colunas: {list(df_check.columns)[:10]}")

print(f"\n{'='*80}")
print(f"✅ PREPARAÇÃO COMPLETA!")
print(f"{'='*80}")
print(f"\n🎯 Próximo passo: Use '{OUTPUT_CSV}' para treinar o modelo!")