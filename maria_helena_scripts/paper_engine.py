"""
Paper Trading Engine - Simulador de ordens
"""
import time
from typing import Dict, Any

class PaperEngine:
    def __init__(self, fee_rate: float = 0.0010, slippage_bp: int = 5):
        """
        Inicializa o simulador de paper trading
        
        Args:
            fee_rate: Taxa de fee (0.0010 = 0.1%)
            slippage_bp: Slippage em basis points (5 = 0.05%)
        """
        self.fee_rate = float(fee_rate)
        self.slippage_bp = int(slippage_bp)
        self.orders = []
    
    def simulate_order(self, side: str, qty: float, price: float, kind: str = "market") -> Dict[str, Any]:
        """
        Simula execução de uma ordem
        
        Args:
            side: 'buy' ou 'sell'
            qty: Quantidade
            price: Preço
            kind: Tipo de ordem (market, limit)
        
        Returns:
            Dict com detalhes da ordem executada
        """
        # Calcular slippage
        slip = price * (self.slippage_bp / 10000.0)
        
        # Aplicar slippage baseado no lado
        if side.lower() == "buy":
            exec_price = price + slip  # Compra mais caro
        else:
            exec_price = price - slip  # Vende mais barato
        
        # Calcular fee
        fee = abs(exec_price * qty) * self.fee_rate
        
        # Criar ordem
        order = {
            "ts": int(time.time()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "side": side.upper(),
            "qty": float(qty),
            "price": float(price),
            "exec_price": float(exec_price),
            "fee": float(fee),
            "kind": kind,
            "simulated": True,
            "status": "filled"
        }
        
        # Salvar ordem
        self.orders.append(order)
        
        return order
    
    def get_orders(self, limit: int = 10) -> list:
        """Retorna últimas ordens"""
        return self.orders[-limit:]
    
    def get_total_fees(self) -> float:
        """Retorna total de fees pagos"""
        return sum(o.get('fee', 0) for o in self.orders)
