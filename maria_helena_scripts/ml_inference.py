"""
Maria Helena - ML Predictor Module
Módulo de predição com suporte para LSTM, fallback heurístico e cache.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

import numpy as np

# Logger
logger = logging.getLogger(__name__)

# ==========================================
# GLOBAL STATE
# ==========================================

MODEL: Optional[Any] = None
SCALER: Optional[Any] = None
MODEL_METADATA: Dict[str, Any] = {}
PREDICTION_CACHE: Dict[str, Any] = {}
FALLBACK_MODE: bool = False

# ==========================================
# CONFIGURATION
# ==========================================

class MLConfig:
    """Configuração do módulo ML"""
    # Paths (resolvidos para absolutos)
    _model_path_env = os.getenv("MODEL_PATH", "maria-helena-scripts/maria_helena_lstm_model.h5")
    _scaler_path_env = os.getenv("SCALER_PATH", "maria-helena-scripts/scaler.pkl")
    MODEL_PATH = str(Path(_model_path_env).expanduser().resolve())
    SCALER_PATH = str(Path(_scaler_path_env).expanduser().resolve())

    # Limiares
    BUY_THRESHOLD = float(os.getenv("ML_BUY_THRESHOLD", "0.65"))
    SELL_THRESHOLD = float(os.getenv("ML_SELL_THRESHOLD", "0.35"))

    # Confiança mínima (para filtros futuros)
    MIN_CONFIDENCE = float(os.getenv("ML_MIN_CONFIDENCE", "0.6"))

    # Nº de features esperadas (ajuste para seu vetor real)
    EXPECTED_FEATURES = int(os.getenv("ML_EXPECTED_FEATURES", "7"))

    # Cache
    CACHE_ENABLED = os.getenv("ML_CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("ML_CACHE_TTL", "60"))  # segundos

    # Fallback
    USE_FALLBACK = os.getenv("ML_USE_FALLBACK", "true").lower() == "true"


# ==========================================
# MODEL LOADING
# ==========================================

def _load_model() -> Tuple[Optional[Any], Optional[Any], bool]:
    """
    Carrega o modelo LSTM e scaler, se disponíveis.
    Returns: (model, scaler, success)
    """
    try:
        from tensorflow import keras  # type: ignore
        import pickle  # type: ignore

        model_path = Path(MLConfig.MODEL_PATH)
        scaler_path = Path(MLConfig.SCALER_PATH)

        if not model_path.exists():
            logger.warning(f"Modelo não encontrado: {model_path}")
            return None, None, False

        if not scaler_path.exists():
            logger.warning(f"Scaler não encontrado: {scaler_path}")
            return None, None, False

        logger.info(f"Carregando modelo LSTM de: {model_path}")
        model = keras.models.load_model(str(model_path))

        logger.info(f"Carregando scaler de: {scaler_path}")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        if hasattr(model, "input_shape"):
            logger.info(f"Modelo carregado. Input shape: {getattr(model, 'input_shape', None)}")

        logger.info("Modelo e scaler carregados com sucesso.")
        return model, scaler, True

    except ImportError as e:
        logger.warning(f"TensorFlow/dep. não disponível: {e}")
        return None, None, False
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return None, None, False


def _lazy_load() -> bool:
    """
    Carrega modelo na primeira predição.
    Returns: True se modelo real carregado; False se fallback.
    """
    global MODEL, SCALER, MODEL_METADATA, FALLBACK_MODE

    if MODEL is not None:
        return not FALLBACK_MODE

    model, scaler, success = _load_model()

    if success:
        MODEL = model
        SCALER = scaler
        FALLBACK_MODE = False
        MODEL_METADATA = {
            "loaded_at": datetime.now().isoformat(),
            "model_path": MLConfig.MODEL_PATH,
            "scaler_path": MLConfig.SCALER_PATH,
            "fallback": False,
            "type": "LSTM",
        }
        logger.info("🤖 Usando modelo LSTM treinado")
        return True

    # Fallback
    MODEL = "fallback"
    SCALER = None
    FALLBACK_MODE = True
    MODEL_METADATA = {
        "loaded_at": datetime.now().isoformat(),
        "fallback": True,
        "type": "heuristic",
        "strategy": MLConfig.FALLBACK_STRATEGY,
    }

    if MLConfig.USE_FALLBACK:
        logger.warning("⚠️  Usando estratégia fallback (heurística)")
        return False

    raise RuntimeError("Modelo indisponível e fallback desabilitado")


# ==========================================
# PREDICTION
# ==========================================

def _predict_with_model(features: np.ndarray) -> Dict[str, Any]:
    """
    Predição com modelo LSTM.
    features: array 1D normalizado de tamanho EXPECTED_FEATURES
    """
    try:
        x = features.reshape(1, -1)
        if SCALER is not None:
            x = SCALER.transform(x)

        pred = MODEL.predict(x, verbose=0)  # type: ignore
        score = float(pred[0][0])

        if score > MLConfig.BUY_THRESHOLD:
            action, confidence = "BUY", score
        elif score < MLConfig.SELL_THRESHOLD:
            action, confidence = "SELL", 1.0 - score
        else:
            action, confidence = "HOLD", 0.5

        return {
            "action": action,
            "score": score,
            "confidence": float(np.clip(confidence, 0, 1)),
            "meta": {
                "model": "LSTM",
                "features_dim": int(x.shape[1]),
                "threshold_buy": MLConfig.BUY_THRESHOLD,
                "threshold_sell": MLConfig.SELL_THRESHOLD,
            },
        }

    except Exception as e:
        logger.error(f"Erro na predição com modelo: {e}")
        return _predict_with_heuristic(features)


def _predict_with_heuristic(features: np.ndarray) -> Dict[str, Any]:
    """
    Predição heurística (fallback).
    """
    base = float(np.clip((features.mean() + 1.0) / 2.0, 0.0, 1.0))

    if MLConfig.FALLBACK_STRATEGY == "conservative":
        buy_th, sell_th = 0.75, 0.25
    elif MLConfig.FALLBACK_STRATEGY == "aggressive":
        buy_th, sell_th = 0.55, 0.45
    else:
        buy_th, sell_th = MLConfig.BUY_THRESHOLD, MLConfig.SELL_THRESHOLD

    # Peso extra simples em [rsi, volume, momentum] se existirem
    score = base
    if features.shape[0] >= 3:
        rsi_w = features[0] if features[0] != 0 else 0.5
        vol_w = features[1] if features[1] != 0 else 0.5
        mom_w = features[2] if features[2] != 0 else 0.5
        score = (base * 0.4) + (rsi_w * 0.3) + (vol_w * 0.2) + (mom_w * 0.1)
        score = float(np.clip(score, 0.0, 1.0))

    if score > buy_th:
        action = "BUY"
        confidence = (score - buy_th) / (1.0 - buy_th)
    elif score < sell_th:
        action = "SELL"
        confidence = (sell_th - score) / sell_th
    else:
        action = "HOLD"
        confidence = 0.3

    return {
        "action": action,
        "score": score,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "meta": {
            "model": "heuristic",
            "strategy": MLConfig.FALLBACK_STRATEGY,
            "features_dim": int(features.shape[0]),
            "threshold_buy": buy_th,
            "threshold_sell": sell_th,
            "warning": "fallback ativo — treine/carregue o LSTM para melhor acurácia",
        },
    }


def _get_cache_key(features: np.ndarray) -> str:
    return str(hash(features.tobytes()))


def predict(feature_vector: np.ndarray) -> Dict[str, Any]:
    """
    Predição principal com cache + fallback.
    """
    # Normalizar tipo/formato
    if not isinstance(feature_vector, np.ndarray):
        feature_vector = np.array(feature_vector, dtype=np.float32)
    feature_vector = feature_vector.astype(np.float32, copy=False)

    # Validações não bloqueantes (apenas log)
    if feature_vector.ndim != 1:
        logger.warning(f"Features devem ser 1D; recebido shape={feature_vector.shape}")
    if feature_vector.shape[0] != MLConfig.EXPECTED_FEATURES:
        logger.warning(f"Esperado {MLConfig.EXPECTED_FEATURES} features; recebido {feature_vector.shape[0]}")
    if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
        logger.warning("Features contêm NaN/Inf")

    # Cache
    if MLConfig.CACHE_ENABLED:
        cache_key = _get_cache_key(feature_vector)
        cached = PREDICTION_CACHE.get(cache_key)
        if cached:
            age = (datetime.now() - cached["timestamp"]).total_seconds()
            if age < MLConfig.CACHE_TTL:
                logger.debug(f"Cache hit ({age:.1f}s)")
                return cached["prediction"]

    # Carregar modelo se necessário
    has_model = _lazy_load()

    # Predizer
    if has_model and not FALLBACK_MODE:
        result = _predict_with_model(feature_vector)
    else:
        result = _predict_with_heuristic(feature_vector)

    # Enriquecer
    result["timestamp"] = datetime.now().isoformat()
    result["model_info"] = MODEL_METADATA

    # Cache
    if MLConfig.CACHE_ENABLED:
        PREDICTION_CACHE[cache_key] = {
            "prediction": result,
            "timestamp": datetime.now(),
        }
        # Evitar crescimento
        if len(PREDICTION_CACHE) > 100:
            oldest_key = min(PREDICTION_CACHE.keys(), key=lambda k: PREDICTION_CACHE[k]["timestamp"])
            del PREDICTION_CACHE[oldest_key]

    logger.info(
        f"Predição: {result['action']} "
        f"(score={result['score']:.3f}, conf={result['confidence']:.3f}, fallback={FALLBACK_MODE})"
    )
    return result


# ==========================================
# UTILS
# ==========================================

def get_model_info() -> Dict[str, Any]:
    _lazy_load()
    return MODEL_METADATA


def reload_model() -> bool:
    global MODEL, SCALER, MODEL_METADATA, FALLBACK_MODE
    MODEL = None
    SCALER = None
    MODEL_METADATA = {}
    FALLBACK_MODE = False
    logger.info("Recarregando modelo ML...")
    return _lazy_load()


def clear_cache():
    global PREDICTION_CACHE
    PREDICTION_CACHE = {}
    logger.info("Cache ML limpo")


def get_cache_stats() -> Dict[str, Any]:
    return {
        "size": len(PREDICTION_CACHE),
        "enabled": MLConfig.CACHE_ENABLED,
        "ttl": MLConfig.CACHE_TTL,
    }


def export_config() -> str:
    cfg = {
        "model_path": MLConfig.MODEL_PATH,
        "scaler_path": MLConfig.SCALER_PATH,
        "buy_threshold": MLConfig.BUY_THRESHOLD,
        "sell_threshold": MLConfig.SELL_THRESHOLD,
        "min_confidence": MLConfig.MIN_CONFIDENCE,
        "expected_features": MLConfig.EXPECTED_FEATURES,
        "cache_enabled": MLConfig.CACHE_ENABLED,
        "cache_ttl": MLConfig.CACHE_TTL,
        "use_fallback": MLConfig.USE_FALLBACK,
        "fallback_strategy": MLConfig.FALLBACK_STRATEGY,
        "model_info": MODEL_METADATA,
    }
    return json.dumps(cfg, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print("🧪 Testando ML Predictor\n")
    print("1) Config:\n", export_config(), "\n")

    test_features = np.array([0.6, 0.5, 0.7, 0.4, 0.5, 0.8, 0.3], dtype=np.float32)
    ok_shape = (test_features.ndim == 1) and (test_features.shape[0] == MLConfig.EXPECTED_FEATURES)
    print(f"2) Features válidas? {ok_shape} (shape={test_features.shape})")

    pred = predict(test_features)
    print(f"4) Cache: {get_cache_stats()}")
