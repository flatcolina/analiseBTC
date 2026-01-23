#!/usr/bin/env python3
"""
=============================================================================
ALGORITMO 2: CONFLUÊNCIA COMPLETA PARA SHORT (VENDA)
=============================================================================

DESCRIÇÃO:
Este algoritmo identifica setups de venda (SHORT) baseados na confluência 
completa de todos os indicadores da estratégia. É o espelho do Algoritmo 1,
otimizado para identificar oportunidades de venda em tendências de baixa.

LÓGICA DO PADRÃO:
1. Preço ABAIXO da VWAP (viés de baixa no dia)
2. EMAs alinhadas para baixa: EMA9 < EMA21 < EMA55
3. Preço em pullback (próximo à EMA9 ou entre EMA9 e EMA21)
4. RSI < 50 (força vendedora)
5. MACD < Linha de Sinal (momentum negativo)

APLICAÇÃO:
- Timeframe recomendado: 1 minuto ou 5 minutos
- Par: BTCUSD ou qualquer par de alta liquidez
- Melhor horário: Sessões de alta volatilidade

EXEMPLO DE REFERÊNCIA: Exemplo 3 e 5 do documento
=============================================================================
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class SignalStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class SetupSignal:
    timestamp: str
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    strength: SignalStrength
    score: float
    indicators: Dict
    reason: str

class ConfluenceShortDetector:
    """
    Detector de setups SHORT baseado em confluência completa de indicadores.
    """
    
    def __init__(self, 
                 ema_periods: Tuple[int, int, int] = (9, 21, 55),
                 rsi_period: int = 14,
                 macd_params: Tuple[int, int, int] = (12, 26, 9),
                 atr_period: int = 14,
                 atr_multiplier_sl: float = 1.5,
                 risk_reward_ratio: float = 1.5):
        
        self.ema_fast, self.ema_mid, self.ema_slow = ema_periods
        self.rsi_period = rsi_period
        self.macd_fast, self.macd_slow, self.macd_signal = macd_params
        self.atr_period = atr_period
        self.atr_multiplier_sl = atr_multiplier_sl
        self.risk_reward_ratio = risk_reward_ratio
    
    # =========================================================================
    # FUNÇÕES DE CÁLCULO DE INDICADORES (Idênticas ao Algoritmo 1)
    # =========================================================================
    
    def calculate_ema(self, prices: List[float], period: int) -> List[Optional[float]]:
        if len(prices) < period:
            return [None] * len(prices)
        
        multiplier = 2 / (period + 1)
        ema = [None] * (period - 1)
        sma = sum(prices[:period]) / period
        ema.append(sma)
        
        for i in range(period, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_value)
        
        return ema
    
    def calculate_rsi(self, prices: List[float]) -> List[Optional[float]]:
        period = self.rsi_period
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        deltas = [0] + [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        rsi = [None] * period
        avg_gain = sum(gains[1:period+1]) / period
        avg_loss = sum(losses[1:period+1]) / period
        
        for i in range(period, len(prices)):
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))
            
            if i < len(prices) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i+1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i+1]) / period
        
        return rsi
    
    def calculate_macd(self, prices: List[float]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        ema_fast = self.calculate_ema(prices, self.macd_fast)
        ema_slow = self.calculate_ema(prices, self.macd_slow)
        
        macd_line = []
        for i in range(len(prices)):
            if ema_fast[i] is None or ema_slow[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        macd_values = [v for v in macd_line if v is not None]
        if len(macd_values) < self.macd_signal:
            return macd_line, [None] * len(macd_line)
        
        signal_values = self.calculate_ema(macd_values, self.macd_signal)
        
        signal_line = []
        macd_idx = 0
        for i in range(len(macd_line)):
            if macd_line[i] is None:
                signal_line.append(None)
            else:
                if macd_idx < len(signal_values):
                    signal_line.append(signal_values[macd_idx])
                else:
                    signal_line.append(None)
                macd_idx += 1
        
        return macd_line, signal_line
    
    def calculate_atr(self, candles: List[Dict]) -> List[Optional[float]]:
        if len(candles) < self.atr_period + 1:
            return [None] * len(candles)
        
        tr_list = [None]
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        
        atr = [None] * self.atr_period
        first_atr = sum([t for t in tr_list[1:self.atr_period+1] if t]) / self.atr_period
        atr.append(first_atr)
        
        for i in range(self.atr_period + 1, len(candles)):
            atr_value = (atr[-1] * (self.atr_period - 1) + tr_list[i]) / self.atr_period
            atr.append(atr_value)
        
        return atr
    
    def calculate_vwap(self, candles: List[Dict]) -> List[Optional[float]]:
        vwap = []
        cumulative_tp_vol = 0
        cumulative_vol = 0
        
        for candle in candles:
            typical_price = (candle['high'] + candle['low'] + candle['close']) / 3
            vol = candle.get('volume', 0) or 0
            cumulative_tp_vol += typical_price * vol
            cumulative_vol += vol
            
            if cumulative_vol > 0:
                vwap.append(cumulative_tp_vol / cumulative_vol)
            else:
                vwap.append(candle['close'])
        
        return vwap
    
    # =========================================================================
    # LÓGICA DE DETECÇÃO DO SETUP SHORT
    # =========================================================================
    
    def check_ema_alignment_bearish(self, ema9: float, ema21: float, ema55: float) -> bool:
        """Verifica se as EMAs estão alinhadas para BAIXA"""
        return ema9 < ema21 < ema55
    
    def check_pullback_to_ema_short(self, price: float, ema9: float, ema21: float, 
                                     tolerance: float = 0.002) -> bool:
        """
        Verifica se o preço está em pullback para as EMAs (para SHORT).
        O preço deve estar próximo à EMA9 por BAIXO ou entre EMA9 e EMA21.
        """
        # Preço próximo à EMA9 (dentro da tolerância)
        near_ema9 = abs(price - ema9) / ema9 <= tolerance
        
        # Preço entre EMA9 e EMA21 (para short, EMA9 < EMA21)
        between_emas = ema9 * (1 - tolerance) <= price <= ema21
        
        return near_ema9 or between_emas
    
    def calculate_signal_score(self, indicators: Dict) -> float:
        """Calcula um score de 0-100 baseado na força dos indicadores."""
        score = 0
        
        # VWAP: Preço abaixo = +20 pontos
        if indicators['price'] < indicators['vwap']:
            score += 20
            distance_pct = (indicators['vwap'] - indicators['price']) / indicators['vwap'] * 100
            if distance_pct > 0.1:
                score += min(5, distance_pct * 10)
        
        # EMAs alinhadas para baixa = +25 pontos
        if self.check_ema_alignment_bearish(indicators['ema9'], indicators['ema21'], indicators['ema55']):
            score += 25
        
        # RSI: Entre 30-50 é ideal para short = até +20 pontos
        rsi = indicators['rsi']
        if 30 < rsi < 50:
            score += 20
        elif rsi <= 30:
            score += 10  # Sobrevenda, menos pontos
        elif 50 <= rsi < 60:
            score += 10
        
        # MACD abaixo da linha de sinal = +20 pontos
        if indicators['macd'] < indicators['macd_signal']:
            score += 20
            if indicators.get('macd_crossed_down', False):
                score += 5
        
        # Pullback para EMA = +15 pontos
        if self.check_pullback_to_ema_short(indicators['price'], indicators['ema9'], indicators['ema21']):
            score += 15
        
        return min(100, score)
    
    def get_signal_strength(self, score: float) -> SignalStrength:
        if score >= 85:
            return SignalStrength.VERY_STRONG
        elif score >= 70:
            return SignalStrength.STRONG
        elif score >= 55:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def detect_setup(self, candles: List[Dict], index: int = -1) -> Optional[SetupSignal]:
        """
        Detecta se existe um setup SHORT válido no índice especificado.
        """
        if len(candles) < max(self.ema_slow, self.macd_slow) + 10:
            return None
        
        closes = [c['close'] for c in candles]
        
        ema9 = self.calculate_ema(closes, self.ema_fast)
        ema21 = self.calculate_ema(closes, self.ema_mid)
        ema55 = self.calculate_ema(closes, self.ema_slow)
        rsi = self.calculate_rsi(closes)
        macd_line, signal_line = self.calculate_macd(closes)
        atr = self.calculate_atr(candles)
        vwap = self.calculate_vwap(candles)
        
        i = index if index >= 0 else len(candles) + index
        
        if any(x is None for x in [ema9[i], ema21[i], ema55[i], rsi[i], 
                                    macd_line[i], signal_line[i], atr[i], vwap[i]]):
            return None
        
        current_price = closes[i]
        
        indicators = {
            'price': current_price,
            'vwap': vwap[i],
            'ema9': ema9[i],
            'ema21': ema21[i],
            'ema55': ema55[i],
            'rsi': rsi[i],
            'macd': macd_line[i],
            'macd_signal': signal_line[i],
            'atr': atr[i],
            'macd_crossed_down': (macd_line[i] < signal_line[i] and 
                                  macd_line[i-1] >= signal_line[i-1] if i > 0 else False)
        }
        
        # =====================================================================
        # CONDIÇÕES DE ENTRADA SHORT - TODAS DEVEM SER VERDADEIRAS
        # =====================================================================
        
        conditions = {
            'below_vwap': current_price < vwap[i],
            'emas_bearish': self.check_ema_alignment_bearish(ema9[i], ema21[i], ema55[i]),
            'rsi_bearish': rsi[i] < 50,
            'macd_bearish': macd_line[i] < signal_line[i],
            'pullback': self.check_pullback_to_ema_short(current_price, ema9[i], ema21[i])
        }
        
        if not all(conditions.values()):
            return None
        
        score = self.calculate_signal_score(indicators)
        strength = self.get_signal_strength(score)
        
        # Calcular SL e TP para SHORT
        sl_distance = atr[i] * self.atr_multiplier_sl
        tp_distance = sl_distance * self.risk_reward_ratio
        
        entry_price = current_price
        stop_loss = entry_price + sl_distance  # SL acima para SHORT
        take_profit = entry_price - tp_distance  # TP abaixo para SHORT
        
        reason = f"Confluência SHORT: Preço abaixo VWAP, EMAs alinhadas (9<21<55), RSI={rsi[i]:.1f}, MACD negativo"
        
        return SetupSignal(
            timestamp=candles[i].get('datetime', str(i)),
            signal_type='SHORT',
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            strength=strength,
            score=round(score, 2),
            indicators=indicators,
            reason=reason
        )
    
    def scan_for_setups(self, candles: List[Dict], lookback: int = 10) -> List[SetupSignal]:
        """Escaneia os últimos N candles em busca de setups SHORT."""
        setups = []
        start_idx = max(self.ema_slow + 10, len(candles) - lookback)
        
        for i in range(start_idx, len(candles)):
            setup = self.detect_setup(candles, i)
            if setup:
                setups.append(setup)
        
        return setups


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ALGORITMO 2: CONFLUÊNCIA COMPLETA PARA SHORT")
    print("=" * 70)
    
    detector = ConfluenceShortDetector(
        ema_periods=(9, 21, 55),
        rsi_period=14,
        atr_multiplier_sl=1.5,
        risk_reward_ratio=1.5
    )
    
    print("\nParâmetros configurados:")
    print(f"  - EMAs: {detector.ema_fast}, {detector.ema_mid}, {detector.ema_slow}")
    print(f"  - RSI: {detector.rsi_period} períodos")
    print(f"  - MACD: {detector.macd_fast}, {detector.macd_slow}, {detector.macd_signal}")
    print(f"  - ATR: {detector.atr_period} períodos")
    print(f"  - SL: {detector.atr_multiplier_sl}x ATR")
    print(f"  - TP: {detector.risk_reward_ratio}x Risco")
    
    print("\n" + "=" * 70)
    print("CONDIÇÕES PARA SINAL SHORT:")
    print("=" * 70)
    print("1. Preço < VWAP (viés de baixa)")
    print("2. EMA9 < EMA21 < EMA55 (tendência de baixa)")
    print("3. Preço em pullback (próximo às EMAs)")
    print("4. RSI < 50 (força vendedora)")
    print("5. MACD < Linha de Sinal (momentum negativo)")
    print("\n>>> TODAS as condições devem ser verdadeiras <<<")
