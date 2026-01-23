from typing import List, Optional, Literal
from models import Candle, TradeState, Direction
from indicators import rsi_series, macd_series, ema_series # Importar funções de cálculo de indicadores

class TradeIntelligence:
    """
    Módulo de Inteligência Dinâmica para Gestão de Operações Ativas.
    Toma decisões não-engessadas (Breakeven, Saída Imediata, Trailing)
    baseadas em novas análises de momentum e volatilidade após a entrada.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.BE_TRIGGER_R = cfg.get("BE_TRIGGER_R", 0.8) # Ex: 0.8 R (80% do SL)
        self.TRAIL_MULT_ATR = cfg.get("TRAIL_MULT_ATR", 0.8) # Ex: 0.8 ATR para trailing stop

    def _calculate_pnl_r(self, trade: TradeState, current_price: float) -> float:
        """Calcula o PnL atual em termos de Risco (R)."""
        risk_amount = abs(trade.entry_exec - trade.sl_price)
        current_pnl = abs(current_price - trade.entry_exec)
        
        if risk_amount == 0:
            return 0.0
        
        r_value = current_pnl / risk_amount
        
        # Ajustar o sinal do R
        if trade.direction == "LONG":
            return r_value if current_price >= trade.entry_exec else -r_value
        else: # SHORT
            return r_value if current_price <= trade.entry_exec else -r_value

    def analyze_and_manage(self, trade: TradeState, candles: List[Candle]) -> Literal["HOLD", "MOVE_BE", "TRAIL", "EXIT_IMMEDIATE"]:
        """
        Analisa o estado atual da operação e retorna a ação recomendada.
        A análise é baseada nos últimos candles (movimentação pós-entrada).
        """
        
        if not candles:
            return "HOLD"

        current_candle = candles[-1]
        current_price = current_candle.close
        
        # 1. Análise de Risco (Breakeven)
        pnl_r = self._calculate_pnl_r(trade, current_price)
        
        if pnl_r >= self.BE_TRIGGER_R and not trade.be_moved:
            # Se o trade andou o suficiente (ex: 0.8R) e o Breakeven ainda não foi movido
            # A decisão inteligente é proteger o capital.
            return "MOVE_BE"

        # 2. Análise de Momentum (Saída Imediata ou Trailing)
        
        # Recalcular indicadores nos últimos 14 candles para nova análise
        closes = [c.close for c in candles]
        rsis = rsi_series(candles, 14)
        macd_line, signal_line, histogram = macd_series(closes)
        
        current_rsi = rsis[-1] if rsis and rsis[-1] is not None else 50
        current_macd_hist = histogram[-1] if histogram and histogram[-1] is not None else 0
        
        # a) Saída Imediata (Perda de Momentum Forte / Reversão)
        if trade.direction == "LONG":
            # Sinais de reversão para LONG: RSI em sobrecompra + MACD histograma caindo
            if current_rsi > 70 and current_macd_hist < 0:
                return "EXIT_IMMEDIATE"
            # Sinais de fraqueza: Cruzamento de EMA 9 abaixo da EMA 21
            if current_candle.ema9 is not None and current_candle.ema21 is not None and current_candle.ema9 < current_candle.ema21:
                return "EXIT_IMMEDIATE"
                
        elif trade.direction == "SHORT":
            # Sinais de reversão para SHORT: RSI em sobrevenda + MACD histograma subindo
            if current_rsi < 30 and current_macd_hist > 0:
                return "EXIT_IMMEDIATE"
            # Sinais de fraqueza: Cruzamento de EMA 9 acima da EMA 21
            if current_candle.ema9 is not None and current_candle.ema21 is not None and current_candle.ema9 > current_candle.ema21:
                return "EXIT_IMMEDIATE"

        # b) Trailing Inteligente (Buscar lucros maiores)
        # Se o trade está em lucro (pnl_r > 0) e o momentum ainda é forte (RSI > 60 para LONG ou RSI < 40 para SHORT)
        if pnl_r > 0:
            is_strong_momentum = (trade.direction == "LONG" and current_rsi > 60) or \
                                 (trade.direction == "SHORT" and current_rsi < 40)
            
            if is_strong_momentum:
                # O Trailing Stop é baseado no ATR para ser adaptativo à volatilidade
                return "TRAIL"

        # 3. Default
        return "HOLD"

    def get_new_sl_price(self, trade: TradeState, candles: List[Candle], action: Literal["MOVE_BE", "TRAIL"]) -> float:
        """Calcula o novo preço de Stop Loss baseado na ação."""
        if not candles:
            return trade.sl_price
            
        current_candle = candles[-1]
        current_price = current_candle.close
        atr = current_candle.atr if current_candle.atr is not None else trade.atr_at_entry
        
        if action == "MOVE_BE":
            # Mover para o preço de entrada (Breakeven)
            return trade.entry_exec
            
        elif action == "TRAIL":
            # Trailing Stop: Mover o SL para um ponto seguro baseado no ATR
            trail_distance = atr * self.TRAIL_MULT_ATR
            
            if trade.direction == "LONG":
                # Novo SL deve ser o preço atual menos a distância de trailing
                new_sl = current_price - trail_distance
                # O novo SL nunca deve ser menor que o SL atual (apenas sobe)
                return max(new_sl, trade.sl_price)
            else: # SHORT
                # Novo SL deve ser o preço atual mais a distância de trailing
                new_sl = current_price + trail_distance
                # O novo SL nunca deve ser maior que o SL atual (apenas desce)
                return min(new_sl, trade.sl_price)
                
        return trade.sl_price

# Exemplo de uso (apenas para referência, não será executado aqui)
# intelligence = TradeIntelligence(cfg={"BE_TRIGGER_R": 0.5, "TRAIL_MULT_ATR": 1.0})
# action = intelligence.analyze_and_manage(active_trade, recent_candles)
# if action in ["MOVE_BE", "TRAIL"]:
#     new_sl = intelligence.get_new_sl_price(active_trade, recent_candles, action)
#     active_trade.sl_price = new_sl
#     active_trade.be_moved = True # Marcar que o BE foi movido (se for MOVE_BE)
