"""
Execution Router - Roteamento de ordens
"""
from typing import Dict, Any, Optional

class ExecutionRouter:
    def __init__(self, route: str, paper_engine=None, exchange=None):
        """
        Inicializa o roteador de execução
        
        Args:
            route: 'paper', 'testnet' ou 'real'
            paper_engine: Instância do PaperEngine
            exchange: Instância da exchange (ccxt)
        """
        self.route = (route or "paper").lower()
        self.paper_engine = paper_engine
        self.exchange = exchange
        
        # Validar configuração
        if self.route == "paper" and not paper_engine:
            raise RuntimeError("PaperEngine é obrigatório para rota 'paper'")
        
        if self.route in ("testnet", "real") and not exchange:
            raise RuntimeError("Exchange é obrigatória para rotas 'testnet' ou 'real'")
    
    def send_order(self, side: str, qty: float, price: float, kind: str = "market") -> Dict[str, Any]:
        """
        Envia ordem para execução
        
        Args:
            side: 'buy' ou 'sell'
            qty: Quantidade
            price: Preço
            kind: Tipo de ordem
        
        Returns:
            Dict com resultado da execução
        """
        if self.route == "paper":
            return self.paper_engine.simulate_order(
                side=side, 
                qty=qty, 
                price=price, 
                kind=kind
            )
        
        elif self.route == "testnet":
            # Implementar lógica de testnet
            return {
                "side": side.upper(),
                "qty": qty,
                "exec_price": price,
                "kind": kind,
                "route": "testnet",
                "status": "sent",
                "simulated": False
            }
        
        elif self.route == "real":
            # Implementar lógica de produção (COM MUITO CUIDADO!)
            return {
                "side": side.upper(),
                "qty": qty,
                "exec_price": price,
                "kind": kind,
                "route": "REAL",
                "status": "sent",
                "simulated": False,
                "warning": "REAL MONEY!"
            }
        
        else:
            raise ValueError(f"Rota inválida: {self.route}")
    
    def get_route(self) -> str:
        """Retorna rota atual"""
        return self.route
