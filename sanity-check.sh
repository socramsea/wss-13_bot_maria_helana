#!/usr/bin/env bash
set -euo pipefail

# Config
PROJECT_DIR="${PROJECT_DIR:-/home/sea/Projects/maria-helena-bot}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
VENV_BIN="$VENV_DIR/bin"
ENV_FILE="$PROJECT_DIR/.env"

# Cores
blue() { echo -e "\033[34m$*\033[0m"; }
green() { echo -e "\033[32m$*\033[0m"; }
yellow() { echo -e "\033[33m$*\033[0m"; }
red() { echo -e "\033[31m$*\033[0m"; }

# Helpers

echo
blue "[SANITY] Projeto........: $PROJECT_DIR"

# 1) Projeto e venv
if [[ -x "$VENV_BIN/python" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_BIN/activate"
  green "[OK] venv ativada: $(python -c 'import sys; print(sys.executable)')"
else
  yellow "[WARN] venv não encontrada em $VENV_DIR — tentando Python do sistema"
fi

# 2) TensorFlow
TF_VER="$(python - <<'PY'
try:
  import tensorflow as tf
  print(tf.__version__)
except Exception as e:
  print(f"ERR:{e}")
PY
)"
if [[ "$TF_VER" =~ ^ERR: ]]; then
  yellow "[WARN] TensorFlow indisponível: ${TF_VER#ERR:}"
else
  green "[OK] TensorFlow........: $TF_VER"
fi

# 3) Caminhos do modelo e scaler (.env ou defaults)
[[ -z "${MODEL_PATH}" ]] && MODEL_PATH="$PROJECT_DIR/maria-helena-scripts/maria_helena_lstm_model.h5"
[[ -z "${SCALER_PATH}" ]] && SCALER_PATH="$PROJECT_DIR/maria-helena-scripts/scaler.pkl"


# 4) Import do módulo e predição
python - <<PY
import os, sys, json
import numpy as np

PROJECT_DIR = "$PROJECT_DIR"
paths = [
  os.path.join(PROJECT_DIR, "maria-helena-scripts"),   # legacy
  os.path.join(PROJECT_DIR, "maria_helena_scripts"),   # pacote recomendado
]
for p in paths:
  if p not in sys.path:
    sys.path.append(p)

# Variáveis do modelo (sem sobrescrever se já vierem do ambiente)
os.environ.setdefault("MODEL_PATH", "$MODEL_PATH")
os.environ.setdefault("SCALER_PATH", "$SCALER_PATH")
os.environ.setdefault("ML_EXPECTED_FEATURES", "7")

result = {
  "ok": False,
  "tf": None,
  "import_path": None,
  "fallback": None,
  "model_type": None,
  "action": None,
  "score": None,
  "confidence": None,
  "error": None
}

# TensorFlow
try:
  import tensorflow as tf
  result["tf"] = tf.__version__
except Exception as e:
  result["tf"] = f"ERR:{e}"

# Import do módulo (3 tentativas)
mod = None
try:
  import ml_inference as m
  mod = m; result["import_path"] = "ml_inference (legacy path)"
except Exception:
  try:
    from maria_helena_scripts import ml_inference as m
    mod = m; result["import_path"] = "maria_helena_scripts.ml_inference"
  except Exception as e:
    # Tenta pelo caminho do arquivo (legacy)
    import importlib.util
    path_try = os.path.join(PROJECT_DIR, "maria-helena-scripts", "ml_inference.py")
    try:
      spec = importlib.util.spec_from_file_location("ml_inference", path_try)
      m = importlib.util.module_from_spec(spec)
      assert spec and spec.loader
      spec.loader.exec_module(m)
      mod = m; result["import_path"] = f"file:{path_try}"
    except Exception as e2:
      result["error"] = f"Import fail: {e2!r}"
      print(json.dumps(result))
      raise SystemExit(20)

# Predição
try:
  fv = np.array([0.61,0.52,0.70,0.43,0.50,0.80,0.31], dtype=np.float32)
  info_before = mod.get_model_info()
  pred = mod.predict(fv)
  info_after = pred.get("model_info", {})
  result["fallback"] = bool(info_after.get("fallback", info_before.get("fallback", True)))
  result["model_type"] = info_after.get("type", info_before.get("type"))
  result["action"] = pred.get("action")
  result["score"] = float(pred.get("score"))
  result["confidence"] = float(pred.get("confidence", 0.0))
  result["ok"] = True
  print(json.dumps(result))
  raise SystemExit(0)
except Exception as e:
  result["error"] = f"Predict fail: {e!r}"
  print(json.dumps(result))
  raise SystemExit(30)
PY
rc=$?

# 5) Resultado final
if [[ $rc -eq 0 ]]; then
  green "[OK] Predição executada. Veja acima: fallback=(True/False), action/score/confidence"
else
  red "[FAIL] Sanity-check falhou com código $rc (veja o erro JSON acima)"
fi

exit $rc
