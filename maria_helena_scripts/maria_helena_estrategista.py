#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Maria Helena Estratégista (maria_helena_estrategista.py)
=======================================================

Visão geral
-----------
Implementa o “Estrategista” do robô de trading Maria Helena.

Responsabilidades:
- Conectar-se à exchange via ccxt (com suporte a testnet).
- Buscar e normalizar dados OHLCV (via Normalizer).
- Executar a estratégia RSIVolumeStrategy (gera BUY/SELL/HOLD).
- Aplicar camadas de proteção (RiskManager, TechnicalGuard, CircuitBreaker, CashGate).
- Roteamento e execução de ordens (PaperEngine / ExecutionRouter / OrderManager).
- Coletar e persistir dados padronizados para treino em CSV (modo mentor e offline training).
- Exibir painéis de status no terminal via rich.
- Garantir comportamento seguro por padrão (kill-switch e rota PAPER).

Arquivos gerados / logs:
- Log geral: maria_helena_bot.log
- CSVs de treino: DATA_DIR/ticks_<SYMBOL>_<TIMEFRAME>.csv
  Colunas: ts, symbol, timeframe, price, rsi_norm, volume_norm, price_norm,
           momentum_norm, volatility_norm, ma_fast_norm, ma_slow_norm, action_strategy

Variáveis de ambiente relevantes:
- ABORT_IF_LIVE (true/false; default "true")
  Kill-switch: aborta a execução se detectar modo live/rota real quando não for intencional.
- LIVE_MODE (true/false; default "false")
  Indica intenção de operar ao vivo (usar chaves de API).
- EXECUTION_ROUTE (paper/testnet/real; default "paper")
  Roteamento de ordens (ExecutionRouter).
- DATA_DIR (default: /home/sea/Projects/maria-helena-bot/data/market)
  Diretório onde os CSVs de treino são gravados.
- PAPER_FEE_RATE (float string; default "0.0010")
  Taxa simulada na PaperEngine.
- PAPER_SLIPPAGE_BP (int string; default "5")
  Slippage em basis points para simulação.
- <EXCHANGE>_API_KEY / <EXCHANGE>_SECRET_KEY
  Chaves da exchange (ex.: BINANCE_API_KEY), usadas quando LIVE_MODE=true.

CONFIG esperado (importado de config):
- EXCHANGE, SYMBOL, TIMEFRAME, MENTOR_MODE, LIVE_MODE, CHECK_INTERVAL,
  INITIAL_CAPITAL, OPTIONS, TESTNET, API_KEY, SECRET_KEY, LOOKBACK_PERIOD,
  BOT_NAME, BOT_VERSION.

Segurança e recomendações:
- Padrão é seguro: rota PAPER e kill-switch ativados.
- Nunca versionar chaves; usar env/secrets.
- Auditar proteções antes de LIVE_MODE/rota real.

Licença e segurança operacional:
- Este módulo pode manipular credenciais e enviar ordens reais.
- Exigir revisão de código antes de habilitar operação real.
"""

# Standard library imports
import time
import logging
from typing import Dict, Any, Optional, List
import os
import sys
import csv

# Third-party imports
import ccxt  # type: ignore
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv

# Carrega variáveis de ambiente ANTES do CONFIG
load_dotenv()

# Console rico para UX de terminal
console = Console()

# Logging básico: arquivo + stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("maria_helena_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def assert_not_live() -> None:
    """
    Kill-switch de segurança:
    - Se ABORT_IF_LIVE=true e detectar LIVE_MODE=true ou rota fora de ("paper","testnet"),
      aborta imediatamente a execução para evitar trades reais acidentais.
    - Imprime instruções para correção no .env.
    """
    abort_if_live = os.getenv("ABORT_IF_LIVE", "true").lower() == "true"
    live_mode = os.getenv("LIVE_MODE", "false").lower() == "true"
    route = os.getenv("EXECUTION_ROUTE", "paper").lower()

    if abort_if_live and (live_mode or route not in ("paper", "testnet")):
        print("[KILL-SWITCH] Execução bloqueada (modo live/rota inválida). Ajuste .env e tente novamente.")
        sys.exit(1)

    if route != "paper":
        print(f"[WARN] EXECUTION_ROUTE={route}. Para simulação, use 'paper'.")


# Aplicar kill-switch o mais cedo possível
assert_not_live()


def install_ccxt_safety(exchange: Any) -> Any:
    """
    Endurecimento de segurança para PAPER:
    - Quando LIVE_MODE=false, sobrescreve métodos de criação/edição/cancelamento de ordens
      para lançar RuntimeError se forem chamados (defesa em profundidade).
    - Tenta cobrir endpoints privados cujo nome contenha "order".
    """
    live_mode = os.getenv("LIVE_MODE", "false").lower() == "true"
    if live_mode:
        return exchange

    def _block(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Trade bloqueado: modo PAPER ativo (LIVE_MODE=false)")

    for name in ["create_order", "createOrder", "cancel_order", "cancelOrder", "edit_order", "editOrder"]:
        if hasattr(exchange, name):
            setattr(exchange, name, _block)

    for name in dir(exchange):
        if "order" in name.lower() and name.startswith("private"):
            try:
                setattr(exchange, name, _block)
            except Exception:
                pass

    return exchange


# Tornar diretório do arquivo importável (para módulos locais do projeto)
sys.path.append(os.path.dirname(__file__))

# Módulos locais do projeto (mantidos após load_dotenv)
from paper_engine import PaperEngine
from execution_router import ExecutionRouter
from config import CONFIG
from protection.risk_manager import RiskManager
from protection.technical_guard import TechnicalGuard
from protection.circuit_breaker import CircuitBreaker
from strategies.mentor_signal_processor import MentorSignalProcessor
from data.normalizer import Normalizer
from strategies.rsi_volume_strategy import RSIVolumeStrategy
from protection.cash_gate.cash_gate import CashGate
from core.orders.order_manager import OrderManager


def _append_training_row(signal: Dict[str, Any], symbol: str, timeframe: str) -> None:
    """
    Persistência de amostras para treino (CSV idempotente):
    - Cria DATA_DIR se não existir (default: /home/sea/Projects/maria-helena-bot/data/market).
    - Cria o arquivo com cabeçalho na primeira gravação.
    - Grava 1 linha por ciclo com as 7 features normalizadas e metadados.

    CSV: ticks_<SYMBOL>_<TIMEFRAME>.csv
    Colunas: ts, symbol, timeframe, price, rsi_norm, volume_norm, price_norm,
             momentum_norm, volatility_norm, ma_fast_norm, ma_slow_norm, action_strategy
    """
    try:
        data_dir = os.getenv("DATA_DIR", "/home/sea/Projects/maria-helena-bot/data/market")
        os.makedirs(data_dir, exist_ok=True)

        fname = f"ticks_{symbol.replace('/', '-')}_{timeframe}.csv"
        fpath = os.path.join(data_dir, fname)

        row = {
            "ts": int(signal.get("timestamp", int(time.time() * 1000))),
            "symbol": symbol,
            "timeframe": timeframe,
            "price": float(signal.get("price", 0.0)),
            "rsi_norm": float(signal.get("rsi_norm", 0.0)),
            "volume_norm": float(signal.get("volume_norm", 0.0)),
            "price_norm": float(signal.get("price_norm", 0.0)),
            "momentum_norm": float(signal.get("momentum_norm", 0.0)),
            "volatility_norm": float(signal.get("volatility_norm", 0.0)),
            "ma_fast_norm": float(signal.get("ma_fast_norm", 0.0)),
            "ma_slow_norm": float(signal.get("ma_slow_norm", 0.0)),
            "action_strategy": str(signal.get("action", "HOLD")),
        }

        header = list(row.keys())
        exists = os.path.exists(fpath)
        with open(fpath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        # Falhas de disco/nfs não devem parar o robô
        logger.warning(f"[DATA] falha ao gravar CSV de treino: {e}")


class MariaHelenaBot:
    """
    Orquestração do loop de trading:
    - Inicia exchange, proteções, normalização, estratégia e roteamento.
    - Loop: fetch OHLCV → analisar → gravar CSV → processar sinal → painel → (opcional) atualizar saldo.
    """

    def __init__(self, config: Dict[str, Any] = CONFIG) -> None:
        # Estado e configuração
        self.config: Dict[str, Any] = config
        self.exchange_name: str = config["EXCHANGE"]
        self.symbol: str = config["SYMBOL"]
        self.timeframe: str = config["TIMEFRAME"]
        self.mentor_mode: bool = config["MENTOR_MODE"]
        self.live_mode: bool = config["LIVE_MODE"]
        self.check_interval: int = config["CHECK_INTERVAL"]
        self.capital: float = config["INITIAL_CAPITAL"]
        self.current_balance: float = self.capital  # atualizado em live

        # Exchange e rota
        self.exchange = self._initialize_exchange()
        self.execution_route: str = os.getenv("EXECUTION_ROUTE", "paper").lower()
        self.paper_engine = PaperEngine(
            fee_rate=float(os.getenv("PAPER_FEE_RATE", "0.0010")),
            slippage_bp=int(os.getenv("PAPER_SLIPPAGE_BP", "5"))
        )
        self.router = ExecutionRouter(route=self.execution_route, paper_engine=self.paper_engine)

        # Proteções
        self.risk_manager: RiskManager = RiskManager(self.config)
        self.technical_guard: TechnicalGuard = TechnicalGuard()
        self.circuit_breaker: CircuitBreaker = CircuitBreaker(self.config)
        self.cash_gate: CashGate = CashGate(self.config["INITIAL_CAPITAL"])

        # Normalização e estratégia
        self.normalizer: Normalizer = Normalizer(self.config)
        self.strategy: RSIVolumeStrategy = RSIVolumeStrategy(self.config)

        # Mentor opcional
        self.mentor_processor: Optional[MentorSignalProcessor] = None
        if self.mentor_mode:
            self.mentor_processor = MentorSignalProcessor()

        # Execução real (quando habilitada)
        self.order_manager: OrderManager = OrderManager(
            self.exchange,
            self.risk_manager,
            self.technical_guard,
            self.circuit_breaker,
            self.cash_gate
        )

        self._print_startup_panel()
        logger.info(
            f"Maria Helena Bot inicializado para {self.symbol} ({self.timeframe}) "
            f"no modo {'AO VIVO' if self.live_mode else 'TESTE/PAPER'}."
        )

    def _initialize_exchange(self) -> Any:
        """
        Constrói e configura a instância CCXT:
        - Aplica OPTIONS e testnet (se configurado).
        - Em LIVE_MODE, exige chaves (API/SECRET) e aborta se ausentes.
        - Carrega mercados e instala safety para PAPER (defesa adicional).
        - Em LIVE_MODE, captura saldo inicial.
        """
        exchange_class = getattr(ccxt, self.exchange_name)
        exchange_params: Dict[str, Any] = {
            "enableRateLimit": True,
            "options": self.config["OPTIONS"],
            "newUpdates": True,
        }

        if self.config["TESTNET"]:
            exchange_params["options"]["defaultType"] = "spot"
            if self.exchange_name == "binance":
                exchange_params["urls"] = {
                    "api": "https://testnet.binance.vision/api",
                    "www": "https://testnet.binance.com"
                }
            logger.info(f"Usando Testnet para {self.exchange_name}.")

        api_key = os.getenv(f"{self.exchange_name.upper()}_API_KEY", self.config["API_KEY"])
        secret_key = os.getenv(f"{self.exchange_name.upper()}_SECRET_KEY", self.config["SECRET_KEY"])

        if api_key and secret_key and self.live_mode:
            exchange_params["apiKey"] = api_key
            exchange_params["secret"] = secret_key
        elif self.live_mode:
            logger.warning("Modo AO VIVO ativado, mas API_KEY ou SECRET_KEY não configuradas.")
            console.print("[red]ERRO: API_KEY/SECRET_KEY não configuradas para modo AO VIVO![/red]")
            sys.exit(1)

        exchange = exchange_class(exchange_params)

        try:
            exchange.load_markets()
            exchange = install_ccxt_safety(exchange)

            logger.info(f"Conectado à exchange: {self.exchange_name.upper()} (Testnet: {self.config['TESTNET']})")
            if self.live_mode:
                balance = exchange.fetch_balance()
                self.current_balance = balance["total"][self.symbol.split("/")[1]]
                logger.info(f"Saldo atual na exchange: {self.current_balance} {self.symbol.split('/')[1]}")
        except Exception as e:
            console.print(f"[red]❌ Erro ao conectar/carregar mercados: {e}[/red]")
            logger.error(f"Erro ao conectar/carregar mercados: {e}", exc_info=True)
            sys.exit(1)

        return exchange

    def _fetch_ohlcv(self) -> Optional[List[List[float]]]:
        """
        Busca OHLCV a partir da exchange, tratando falhas comuns:
        - NetworkError/ExchangeError → registra e retorna None (loop prossegue).
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                self.timeframe,
                limit=self.config["LOOKBACK_PERIOD"] + 5
            )
            if not ohlcv:
                console.print(f"[yellow]⚠️ Nenhum dado OHLCV recebido para {self.symbol}.[/yellow]")
                return None
            return ohlcv
        except ccxt.NetworkError as e:
            console.print(f"[red]❌ Erro de rede ao buscar OHLCV: {e}[/red]")
            logger.warning(f"Erro de rede ao buscar OHLCV: {e}")
            return None
        except ccxt.ExchangeError as e:
            console.print(f"[red]❌ Erro da exchange ao buscar OHLCV: {e}[/red]")
            logger.error(f"Erro da exchange ao buscar OHLCV: {e}")
            return None
        except Exception as e:
            console.print(f"[red]❌ Erro inesperado ao buscar OHLCV: {e}[/red]")
            logger.error(f"Erro inesperado ao buscar OHLCV: {e}", exc_info=True)
            return None

    def _get_latest_price(self, ohlcv: List[List[float]]) -> float:
        """Preço de fechamento da última vela (close)."""
        return ohlcv[-1][4]

    def _analyze_strategy(self, ohlcv: List[List[float]]) -> Dict[str, Any]:
        """
        Pipeline de análise:
        - Monta ticker simples a partir da última vela.
        - Normaliza via Normalizer.process(ohlcv, ticker).
        - Converte OHLCV em DataFrame e roda RSIVolumeStrategy.analyze(df).
        - Enriquecimento do sinal: price + campos normalizados.
        - Em mentor_mode: encaminha para MentorSignalProcessor.
        - Em falha: retorna HOLD seguro (não trava o loop).
        """
        try:
            ticker = {
                "last": ohlcv[-1][4],
                "quoteVolume": ohlcv[-1][5],
                "timestamp": ohlcv[-1][0]
            }

            normalized_data = self.normalizer.process(ohlcv, ticker)

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            signal = self.strategy.analyze(df)
            signal["price"] = self._get_latest_price(ohlcv)
            signal.update(normalized_data)

            if self.mentor_mode and self.mentor_processor:
                self.mentor_processor.process_signal(signal)

            logger.info(f"SINAL GERADO: {signal}")
            return signal

        except Exception as e:
            console.print(f"[red]❌ Erro na análise da estratégia: {e}[/red]")
            logger.error(f"Erro na análise da estratégia: {e}", exc_info=True)
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "price": 0.0,
                "reason": f"Erro na análise: {e}"
            }

    def _process_signal(self, signal: Dict[str, Any]) -> None:
        """
        Pipeline de execução do sinal:
        1) Ignora HOLD.
        2) Valida preço.
        3) Calcula tamanho da posição (RiskManager).
        4) Valida e reserva capital (CashGate).
        5) PAPER: ExecutionRouter/PaperEngine. LIVE: OrderManager.
        6) Em falhas, libera reserva e registra erro.
        """
        action = signal.get("action", "HOLD")
        if action == "HOLD":
            return

        current_price = signal.get("price", 0.0)
        if current_price <= 0:
            logger.error("Preço inválido no sinal")
            return

        position_size_usd = self.risk_manager.calculate_position_size(
            self.current_balance,
            signal
        )
        if position_size_usd <= 0:
            logger.warning("Tamanho de posição calculado é zero ou negativo")
            return

        can_reserve, reason = self.cash_gate.can_reserve(position_size_usd)
        if not can_reserve:
            console.print(f"[yellow]🚫 CashGate bloqueou: {reason}[/yellow]")
            logger.warning(f"🚫 CashGate bloqueou: {reason}")
            return

        if not self.cash_gate.reserve(position_size_usd):
            logger.error("Falha ao reservar capital no CashGate")
            return

        try:
            qty = position_size_usd / current_price

            if not self.live_mode or self.execution_route == "paper":
                result = self.router.send_order(
                    side="buy" if action.upper() == "BUY" else "sell",
                    qty=qty,
                    price=current_price,
                    kind="market"
                )
                console.print(f"[green]✅ (PAPER) Ordem simulada: {result}[/green]")
                logger.info(f"(PAPER) Ordem simulada: {result}")
                return

            order = self.order_manager.execute_order(
                action=action,
                amount=qty,
                price=current_price,
                signal=signal
            )

            if order:
                console.print(f"[green]✅ Ordem executada: {order.get('id', 'N/A')}[/green]")
                logger.info(f"Ordem executada com sucesso: {order}")
            else:
                self.cash_gate.release(position_size_usd)
                console.print("[red]❌ Falha ao executar ordem[/red]")
                logger.error("OrderManager retornou None")

        except Exception as e:
            self.cash_gate.release(position_size_usd)
            logger.error(f"Erro ao executar ordem: {e}", exc_info=True)
            console.print(f"[red]❌ Erro: {e}[/red]")

    def _print_startup_panel(self) -> None:
        """Painel inicial com metadados do bot e ambiente."""
        panel_content = Text(justify="center")
        panel_content.append(f"{self.config['BOT_NAME']} v{self.config['BOT_VERSION']}\n", style="bold green")
        panel_content.append(f"Par: [cyan]{self.symbol}[/cyan] | Intervalo: [cyan]{self.timeframe}[/cyan]\n", style="white")
        panel_content.append(f"Exchange: [yellow]{self.exchange_name.upper()}[/yellow] | Testnet: [yellow]{self.config['TESTNET']}[/yellow]\n", style="white")
        panel_content.append(f"Capital Inicial: [green]${self.capital:,.2f}[/green]\n", style="white")
        panel_content.append(f"Modo Mentor: {'[bold magenta]ATIVO[/bold magenta]' if self.mentor_mode else '[dim]DESATIVADO[/dim]'}\n", style="white")
        panel_content.append("\nIniciando operação...\n", style="italic blue")
        console.print(Panel(panel_content, title="🤖 Maria Helena Trading Bot - Status", border_style="bold green"))

    def _print_status_panel(self, current_price: float) -> None:
        """Painel periódico de status: preço, posição, P&L e circuit breaker."""
        panel_content = Text(justify="center")
        panel_content.append(f"Par: [cyan]{self.symbol}[/cyan] | Intervalo: [cyan]{self.timeframe}[/cyan]\n", style="white")
        panel_content.append(f"Preço Atual: [bold green]${current_price:,.2f}[/bold green]\n", style="white")

        position_info = self.order_manager.get_position_info()
        if position_info["is_in_position"]:
            panel_content.append(
                f"Posição: [yellow]ABERTA[/yellow] | "
                f"Entrada: [yellow]${position_info['entry_price']:,.2f}[/yellow] | "
                f"Quantidade: [yellow]{position_info['amount']:.4f} {self.symbol.split('/')[0]}[/yellow]\n",
                style="white"
            )
            unrealized_pnl = (current_price - position_info["entry_price"]) * position_info["amount"]
            pnl_style = "green" if unrealized_pnl >= 0 else "red"
            panel_content.append(f"P&L Não Realizado: [{pnl_style}]${unrealized_pnl:,.2f}[/{pnl_style}]\n", style="white")
        else:
            panel_content.append("[dim]Posição: FECHADA[/dim]\n", style="white")

        circuit_breaker_status = f"{self.circuit_breaker.current_losses}/{self.circuit_breaker.max_losses}"
        if self.circuit_breaker.is_tripped:
            panel_content.append(f"Circuit Breaker: [bold red]ATIVADO[/bold red] ({circuit_breaker_status})\n", style="white")
        else:
            panel_content.append(f"Circuit Breaker: [green]OK[/green] ({circuit_breaker_status})\n", style="white")

        console.print(Panel(panel_content, title="⚙️ Status Atual do Bot", border_style="dim"))

    def run(self) -> None:
        """
        Loop principal:
        - Busca OHLCV → analisa estratégia → grava CSV de treino → processa sinal → painel.
        - Atualiza saldo em LIVE (se disponível) e trata erros sem interromper execução.
        """
        while True:
            try:
                ohlcv_data = self._fetch_ohlcv()
                if ohlcv_data:
                    # 1) Analisar estratégia
                    signal = self._analyze_strategy(ohlcv_data)

                    # 2) Persistir amostra de treino (não para execução)
                    try:
                        _append_training_row(signal, self.symbol, self.timeframe)
                    except Exception as e:
                        logger.warning(f"[DATA] falha ao gravar CSV de treino (run): {e}")

                    # 3) Processar sinal (paper/live)
                    self._process_signal(signal)

                    # 4) Painel
                    current_price = self._get_latest_price(ohlcv_data)
                    self._print_status_panel(current_price)

                # Atualiza saldo quando em live (se API suportar)
                if self.live_mode and hasattr(self.exchange, "fetch_balance"):
                    try:
                        balance = self.exchange.fetch_balance()
                        self.current_balance = balance["total"][self.symbol.split("/")[1]]
                    except Exception as e:
                        logger.warning(f"Erro ao atualizar saldo: {e}")

            except ccxt.NetworkError as e:
                console.print(f"[red]❌ Erro de rede: {e}[/red]")
                logger.warning(f"Erro de rede, tentando novamente em {self.check_interval}s. {e}")
            except ccxt.ExchangeError as e:
                console.print(f"[red]❌ Erro da exchange: {e}[/red]")
                logger.error(f"Erro da exchange, tentando novamente em {self.check_interval}s. {e}")
            except Exception as e:
                console.print(f"[red]❌ Erro inesperado no loop principal: {e}[/red]")
                logger.exception("Erro não tratado no loop principal")

            console.print(f"\n[dim]Aguardando {self.check_interval} segundos...[/dim]")
            time.sleep(self.check_interval)


def main() -> None:
    """Ponto de entrada do bot com tratamento de encerramento e erros fatais."""
    try:
        bot = MariaHelenaBot()
        bot.run()
    except KeyboardInterrupt:
        console.print("\n[cyan]🛑 Bot interrompido pelo usuário.[/cyan]")
        logger.info("Bot interrompido pelo usuário.")
    except Exception as e:
        console.print(f"[red]❌ Erro fatal: {e}[/red]")
        logger.exception("Erro fatal não tratado no main.")
    finally:
        console.print("\n[bold blue]🚀 Maria Helena Bot encerrado.[/bold blue]")


if __name__ == "__main__":
    main()