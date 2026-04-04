#!/usr/bin/env bash
set -euo pipefail

# Minimal install for Maria Helena Bot (PAPER + OKX)
PROJECT_DIR="/home/sea/Projects/maria-helena-bot"
SERVICE_NAME="maria-helena"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
PIP_BIN="${PROJECT_DIR}/venv/bin/pip"

echo "[1/4] Preparando diretório do projeto: ${PROJECT_DIR}"
if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "Erro: diretório do projeto não encontrado: ${PROJECT_DIR}"
    exit 1
fi

echo "[2/4] venv + dependências"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    python3 -m venv "${PROJECT_DIR}/venv"
fi
# shellcheck disable=SC1090
source "${PROJECT_DIR}/venv/bin/activate"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel >/dev/null
"${PIP_BIN}" install "ccxt>=4.0.0" pandas rich python-dotenv requests >/dev/null

echo "[3/4] .env seguro (PAPER) + OKX"
ENV_FILE="${PROJECT_DIR}/.env"
touch "${ENV_FILE}"
# Zera chaves essenciais e regrava
cat >> "${ENV_FILE}" << 'EOF'
EXCHANGE=okx
TESTNET=false
LIVE_MODE=false
EXECUTION_ROUTE=paper
ABORT_IF_LIVE=true
LOG_EXECUTIONS=true
PAPER_FEE_RATE=0.0010
PAPER_SLIPPAGE_BP=5
EOF

echo "[3.1/4] Criando/atualizando serviço systemd"
sudo tee "${UNIT_FILE}" > /dev/null <<EOF
[Unit]
Description=Maria Helena Bot (Paper)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=sea
Group=sea
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=${PROJECT_DIR}/.env
Environment=LIVE_MODE=false
Environment=EXECUTION_ROUTE=paper
Environment=ABORT_IF_LIVE=true
Environment=LOG_EXECUTIONS=true
ExecStart=${PYTHON_BIN} -u ./maria-helena-scripts/maria_helena_estrategista.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

echo "[4/4] Ativando serviço"
sudo systemctl daemon-reload
sudo systemctl restart "${SERVICE_NAME}"

echo "---------------------------------------------"
echo "Status e últimos logs:"
echo "---------------------------------------------"
echo "Pronto. Bot em PAPER com dados reais (OKX)."
