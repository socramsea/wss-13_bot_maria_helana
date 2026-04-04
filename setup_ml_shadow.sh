#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG =========
PROJECT_DIR="/home/sea/Projects/maria-helena-bot"
ENV_FILE="${PROJECT_DIR}/.env"
VENV_DIR="${PROJECT_DIR}/venv"
VENV_BIN="${VENV_DIR}/bin"
ESTRATEGISTA="${PROJECT_DIR}/maria-helena-scripts/maria_helena_estrategista.py"
ML_MODULE="${PROJECT_DIR}/maria-helena-scripts/ml_inference.py"
SERVICE_NAME="maria-helena"

# Modelo já treinado (ajuste se necessário)
MODEL_PATH="${PROJECT_DIR}/maria-helena-scripts/maria_helena_lstm_model.h5"
SCALER_PATH="${PROJECT_DIR}/maria-helena-scripts/scaler.pkl"

# ML params
ML_EXPECTED_FEATURES="7"
ML_BUY_THRESHOLD="0.65"
ML_SELL_THRESHOLD="0.35"
ML_USE_FALLBACK="true"
ML_FALLBACK_STRATEGY="conservative"
ML_SHADOW_LOG_LIMIT="10"

# ========= FUNÇÕES =========
info() { echo "[INFO] $*"; }
ok()   { echo "[OK]   $*"; }
warn() { echo "[WARN] $*"; }
err()  { echo "[ERR]  $*"; }

require_cmd() {
}

upsert_env() {
  local key="$1"; local val="$2"
  touch "${ENV_FILE}"
  if grep -qE "^${key}=" "${ENV_FILE}"; then
    # atualiza
  else
    # adiciona
    printf "%s=%s\n" "${key}" "${val}" >> "${ENV_FILE}"
  fi
}

write_ml_module() {
  mkdir -p "$(dirname "${ML_MODULE}")"
  cat > "${ML_MODULE}" << 'PY'
"""
Maria Helena - ML Predictor Module (LSTM + fallback + cache)
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)

MODEL: Optional[Any] = None  # type: ignore
SCALER: Optional[Any] = None  # type: ignore
MODEL_METADATA: Dict[str, Any] = {}
PREDICTION_CACHE: Dict[str, Any] = {}
FALLBACK_MODE: bool = False

class MLConfig:
    _model_path_env = os.getenv("MODEL_PATH", "maria-helena-scripts/maria_helena_lstm_model.h5")
    _scaler_path_env = os.getenv("SCALER_PATH", "maria-helena-scripts/scaler.pkl")
    MODEL_PATH = str(Path(_model_path_env).expanduser().resolve())
    SCALER_PATH = str(Path(_scaler_path_env).expanduser().resolve())
    BUY_THRESHOLD = float(os.getenv("ML_BUY_THRESHOLD", "0.65"))
    SELL_THRESHOLD = float(os.getenv("ML_SELL_THRESHOLD", "0.35"))
    EXPECTED_FEATURES = int(os.getenv("ML_EXPECTED_FEATURES", "7"))
    CACHE_ENABLED = os.getenv("ML_CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("ML_CACHE_TTL", "60"))
    USE_FALLBACK = os.getenv("ML_USE_FALLBACK", "true").lower() == "true"
    FALLBACK_STRATEGY = os.getenv("ML_FALLBACK_STRATEGY", "conservative")

def _load_model() -> Tuple[Optional[Any], Optional[Any], bool]:  # type: ignore
    try:
        from tensorflow import keras  # type: ignore
        import pickle  # type: ignore
        mp = Path(MLConfig.MODEL_PATH); sp = Path(MLConfig.SCALER_PATH)
        if not mp.exists() or not sp.exists():
            logger.warning(f"Modelo/scaler não encontrados: {mp} | {sp}")
            return None, None, False
        logger.info(f"Carregando modelo LSTM de: {mp}")
        model = keras.models.load_model(str(mp))
        logger.info(f"Carregando scaler de: {sp}")
        with open(sp, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler, True
    except ImportError as e:
        logger.warning(f"TensorFlow ausente: {e}")
        return None, None, False
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return None, None, False

def _lazy_load() -> bool:
    global MODEL, SCALER, MODEL_METADATA, FALLBACK_MODE
    if MODEL is not None:
        return not FALLBACK_MODE
    model, scaler, ok = _load_model()
    if ok:
        MODEL, SCALER, FALLBACK_MODE = model, scaler, False
        MODEL_METADATA = {"loaded_at": datetime.now().isoformat(), "model_path": MLConfig.MODEL_PATH, "scaler_path": MLConfig.SCALER_PATH, "fallback": False, "type": "LSTM"}
        logger.info("🤖 Usando modelo LSTM treinado")
        return True
    MODEL, SCALER, FALLBACK_MODE = "fallback", None, True
    MODEL_METADATA = {"loaded_at": datetime.now().isoformat(), "fallback": True, "type": "heuristic", "strategy": MLConfig.FALLBACK_STRATEGY}
    if MLConfig.USE_FALLBACK:
        logger.warning("⚠️  Usando estratégia fallback (heurística)")
        return False
    raise RuntimeError("Modelo indisponível e fallback desabilitado")

def _predict_with_model(features: np.ndarray) -> Dict[str, Any]:
    try:
        x = features.reshape(1, -1)
        if SCALER is not None:
            x = SCALER.transform(x)  # type: ignore
        pred = MODEL.predict(x, verbose=0)  # type: ignore
        score = float(pred[0][0])
        if score > MLConfig.BUY_THRESHOLD:
            action, conf = "BUY", score
        elif score < MLConfig.SELL_THRESHOLD:
            action, conf = "SELL", 1.0 - score
        else:
            action, conf = "HOLD", 0.5
        return {"action": action, "score": score, "confidence": float(max(0.0, min(1.0, conf))),
                "meta": {"model": "LSTM", "features_dim": int(x.shape[1]),
                         "threshold_buy": MLConfig.BUY_THRESHOLD, "threshold_sell": MLConfig.SELL_THRESHOLD}}
    except Exception as e:
        logger.error(f"Erro na predição com modelo: {e}")
        return _predict_with_heuristic(features)

def _predict_with_heuristic(features: np.ndarray) -> Dict[str, Any]:
    base = float(np.clip((features.mean() + 1.0) / 2.0, 0.0, 1.0))
    if MLConfig.FALLBACK_STRATEGY == "conservative":
        buy_th, sell_th = 0.75, 0.25
    elif MLConfig.FALLBACK_STRATEGY == "aggressive":
        buy_th, sell_th = 0.55, 0.45
    else:
        buy_th, sell_th = MLConfig.BUY_THRESHOLD, MLConfig.SELL_THRESHOLD
    score = base
    if features.shape[0] >= 3:
        rsi_w = features[0] if features[0] != 0 else 0.5
        vol_w = features[1] if features[1] != 0 else 0.5
        mom_w = features[2] if features[2] != 0 else 0.5
        score = float(np.clip((base * 0.4) + (rsi_w * 0.3) + (vol_w * 0.2) + (mom_w * 0.1), 0.0, 1.0))
    if score > buy_th:
        action, conf = "BUY", (score - buy_th) / (1.0 - buy_th)
    elif score < sell_th:
        action, conf = "SELL", (sell_th - score) / sell_th
    else:
        action, conf = "HOLD", 0.3
    return {"action": action, "score": score, "confidence": float(max(0.0, min(1.0, conf))),
            "meta": {"model": "heuristic", "strategy": MLConfig.FALLBACK_STRATEGY,
                     "features_dim": int(features.shape[0]),
                     "threshold_buy": buy_th, "threshold_sell": sell_th,
                     "warning": "fallback ativo — treine/carregue o LSTM para melhor acurácia"}}

def _get_cache_key(features: np.ndarray) -> str:
    return str(hash(features.tobytes()))

def predict(feature_vector: np.ndarray) -> Dict[str, Any]:
    if not isinstance(feature_vector, np.ndarray):
        feature_vector = np.array(feature_vector, dtype=np.float32)
    feature_vector = feature_vector.astype(np.float32, copy=False)
    if feature_vector.ndim != 1:
        logger.warning(f"Features devem ser 1D; shape={feature_vector.shape}")
    if feature_vector.shape[0] != MLConfig.EXPECTED_FEATURES:
        logger.warning(f"Esperado {MLConfig.EXPECTED_FEATURES} features; recebido {feature_vector.shape[0]}")
    if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
        logger.warning("Features contêm NaN/Inf")
    use_cache = MLConfig.CACHE_ENABLED
    key = None
    if use_cache:
        key = _get_cache_key(feature_vector)
        cached = PREDICTION_CACHE.get(key)
        if cached:
            age = (datetime.now() - cached["timestamp"]).total_seconds()
            if age < MLConfig.CACHE_TTL:
                return cached["prediction"]
    has_model = _lazy_load()
    result = _predict_with_model(feature_vector) if has_model and not FALLBACK_MODE else _predict_with_heuristic(feature_vector)
    result["timestamp"] = datetime.now().isoformat()
    result["model_info"] = MODEL_METADATA
    if use_cache and key is not None:
        PREDICTION_CACHE[key] = {"prediction": result, "timestamp": datetime.now()}
        if len(PREDICTION_CACHE) > 100:
            oldest = min(PREDICTION_CACHE.keys(), key=lambda k: PREDICTION_CACHE[k]["timestamp"])
            del PREDICTION_CACHE[oldest]
    logger.info(f"Predição: {result['action']} (score={result['score']:.3f}, conf={result.get('confidence',0.0):.3f}, fallback={FALLBACK_MODE})")
    return result
PY
}

inject_shadow() {
  # imports + vars (se ausentes)
  if ! grep -q "from ml_inference import predict as ml_predict" "${ESTRATEGISTA}"; then
    sed -i '/from dotenv import load_dotenv/a \
import numpy as np\nfrom ml_inference import predict as ml_predict\nML_SHADOW_LOG_LIMIT = int(os.getenv("ML_SHADOW_LOG_LIMIT", "10"))\n_ml_shadow_log_count = 0\n' "${ESTRATEGISTA}"
    ok "Imports ML adicionados"
  else
    ok "Imports ML já presentes"
  fi

  # bloco shadow (se ausente)
  if ! grep -q "===== ML (shadow) =====" "${ESTRATEGISTA}"; then
    sed -i '/signal = self._analyze_strategy(ohlcv_data)/a \
                    # ===== ML (shadow) =====\n\
                    try:\n\
                        fv = np.array([\n\
                            float(signal.get("rsi_norm", 0.0)),\n\
                            float(signal.get("volume_norm", 0.0)),\n\
                            float(signal.get("price_norm", 0.0)),\n\
                            float(signal.get("momentum_norm", 0.0)),\n\
                            float(signal.get("volatility_norm", 0.0)),\n\
                            float(signal.get("ma_fast_norm", 0.0)),\n\
                            float(signal.get("ma_slow_norm", 0.0)),\n\
                        ], dtype=np.float32)\n\
                        ml_out = ml_predict(fv)\n\
                        global _ml_shadow_log_count\n\
                        if _ml_shadow_log_count < ML_SHADOW_LOG_LIMIT:\n\
                            logger.info(f"[ML] shadow: action={ml_out[\'action\']} score={ml_out[\'score\']:.3f} conf={ml_out.get(\'confidence\',0.0):.2f} fallback={ml_out.get(\'model_info\',{}).get(\'fallback\')}")\n\
                            _ml_shadow_log_count += 1\n\
                        signal[\"ml_action\"] = ml_out[\"action\"]\n\
                        signal[\"ml_score\"] = float(ml_out[\"score\"])\n\
                        signal[\"ml_confidence\"] = float(ml_out.get(\"confidence\", 0.0))\n\
                    except Exception as e:\n\
                        logger.warning(f\"[ML] shadow falhou: {e}\")\n\
                    # ===== fim ML (shadow) =====\n' "${ESTRATEGISTA}"
    ok "Bloco shadow injetado"
  else
    ok "Bloco shadow já existe"
  fi
}

# ========= EXECUÇÃO =========
require_cmd sed
require_cmd grep
require_cmd python3

info "Validando estrutura..."
  warn "Modelo/scaler não encontrados. Será usado fallback heurístico (seguro)."
else
  ok "Modelo/scaler encontrados."
fi

info "Venv e deps mínimas..."
if [[ ! -x "${VENV_BIN}/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_BIN}/activate"
python -m pip install --upgrade pip >/dev/null
pip install numpy >/dev/null
ok "Venv OK"

info "Atualizando .env (ML)..."
upsert_env "MODEL_PATH" "${MODEL_PATH}"
upsert_env "SCALER_PATH" "${SCALER_PATH}"
upsert_env "ML_EXPECTED_FEATURES" "${ML_EXPECTED_FEATURES}"
upsert_env "ML_BUY_THRESHOLD" "${ML_BUY_THRESHOLD}"
upsert_env "ML_SELL_THRESHOLD" "${ML_SELL_THRESHOLD}"
upsert_env "ML_USE_FALLBACK" "${ML_USE_FALLBACK}"
upsert_env "ML_FALLBACK_STRATEGY" "${ML_FALLBACK_STRATEGY}"
upsert_env "ML_SHADOW_LOG_LIMIT" "${ML_SHADOW_LOG_LIMIT}"
ok ".env OK"

info "Escrevendo ml_inference.py..."
write_ml_module
ok "ml_inference.py OK"

info "Backup do estrategista e injeção ML..."
cp "${ESTRATEGISTA}" "${ESTRATEGISTA}.bak.$(date +%s)"
inject_shadow

info "Reiniciando serviço..."
sudo systemctl restart "${SERVICE_NAME}"

info "Verificando status..."
sleep 2
echo
ok "Concluído. Veja logs com: sudo journalctl -u ${SERVICE_NAME} -n 120 --no-pager"
