#!/usr/bin/env python3
"""
=============================================================================
ALGORITMO 3: DETECTOR DE MOMENTUM (RSI + MACD)
=============================================================================

DESCRIÇÃO:
Este algoritmo foca especificamente na análise de momentum, utilizando RSI e 
MACD como indicadores principais. É mais sensível a mudanças de momentum e 
pode gerar sinais mais cedo que o algoritmo de confluência completa.

LÓGICA DO PADRÃO:
1. RSI cruzando níveis-chave (30, 50, 70)
2. MACD cruzando a linha de sinal
3. Divergências entre preço e RSI (avançado)
4. Histograma do MACD mudando de direção

DIFERENCIAL:
- Detecta mudanças de momentum ANTES da confirmação de tendência
- Útil para entradas mais agressivas
- Pode identificar reversões potenciais

APLICAÇÃO:
- Timeframe: 1 a 5 minutos
- Melhor em mercados com volatilidade moderada a alta
- Combinar com análise de suporte/resistência

EXEMPLO DE REFERÊNCIA: Usado para confirmar força nos Exemplos 1, 3, 4, 5
=============================================================================
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class MomentumState(Enum):
    """Estados possíveis do momentum"""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

class SignalType(Enum):
    """Tipos de sinais de momentum"""
    RSI_OVERSOLD_EXIT = "rsi_oversold_exit"
    RSI_OVERBOUGHT_EXIT = "rsi_overbought_exit"
    RSI_CENTERLINE_CROSS_UP = "rsi_centerline_cross_up"
    RSI_CENTERLINE_CROSS_DOWN = "rsi_centerline_cross_down"
    MACD_BULLISH_CROSS = "macd_bullish_cross"
    MACD_BEARISH_CROSS = "macd_bearish_cross"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    HISTOGRAM_REVERSAL_UP = "histogram_reversal_up"
    HISTOGRAM_REVERSAL_DOWN = "histogram_reversal_down"

@dataclass
class MomentumSignal:
    """Estrutura de dados para sinais de momentum"""
    timestamp: str
    signal_type: SignalType
    direction: str  # 'LONG' ou 'SHORT'
    strength: float  # 0-100
    rsi_value: float
    macd_value: float
    macd_signal: float
    macd_histogram: float
    description: str

class MomentumDetector:
    """
    Detector de sinais baseado em análise de momentum.
    Foca em RSI e MACD para identificar mudanças de força no mercado.
    """
    
    def __init__(self,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 divergence_lookback: int = 14):
        
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.divergence_lookback = divergence_lookback
    
    # =========================================================================
    # CÁLCULO DE INDICADORES
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
    
    def calculate_macd(self, prices: List[float]) -> Tuple[List, List, List]:
        """Retorna MACD line, Signal line, e Histogram"""
        ema_fast = self.calculate_ema(prices, self.macd_fast)
        ema_slow = self.calculate_ema(prices, self.macd_slow)
        
        macd_line = []
        for i in range(len(prices)):
            if ema_fast[i] is None or ema_slow[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        macd_values = [v for v in macd_line if v is not None]
        if len(macd_values) < self.macd_signal_period:
            return macd_line, [None] * len(macd_line), [None] * len(macd_line)
        
        signal_values = self.calculate_ema(macd_values, self.macd_signal_period)
        
        signal_line = []
        histogram = []
        macd_idx = 0
        
        for i in range(len(macd_line)):
            if macd_line[i] is None:
                signal_line.append(None)
                histogram.append(None)
            else:
                if macd_idx < len(signal_values) and signal_values[macd_idx] is not None:
                    signal_line.append(signal_values[macd_idx])
                    histogram.append(macd_line[i] - signal_values[macd_idx])
                else:
                    signal_line.append(None)
                    histogram.append(None)
                macd_idx += 1
        
        return macd_line, signal_line, histogram
    
    # =========================================================================
    # DETECÇÃO DE SINAIS DE MOMENTUM
    # =========================================================================
    
    def detect_rsi_signals(self, rsi: List[Optional[float]], index: int) -> List[SignalType]:
        """Detecta sinais baseados no RSI"""
        signals = []
        
        if index < 1 or rsi[index] is None or rsi[index-1] is None:
            return signals
        
        current_rsi = rsi[index]
        prev_rsi = rsi[index - 1]
        
        # RSI saindo da zona de sobrevenda (< 30 -> > 30)
        if prev_rsi <= self.rsi_oversold and current_rsi > self.rsi_oversold:
            signals.append(SignalType.RSI_OVERSOLD_EXIT)
        
        # RSI saindo da zona de sobrecompra (> 70 -> < 70)
        if prev_rsi >= self.rsi_overbought and current_rsi < self.rsi_overbought:
            signals.append(SignalType.RSI_OVERBOUGHT_EXIT)
        
        # RSI cruzando a linha central (50) para cima
        if prev_rsi <= 50 and current_rsi > 50:
            signals.append(SignalType.RSI_CENTERLINE_CROSS_UP)
        
        # RSI cruzando a linha central (50) para baixo
        if prev_rsi >= 50 and current_rsi < 50:
            signals.append(SignalType.RSI_CENTERLINE_CROSS_DOWN)
        
        return signals
    
    def detect_macd_signals(self, macd: List, signal: List, histogram: List, 
                            index: int) -> List[SignalType]:
        """Detecta sinais baseados no MACD"""
        signals = []
        
        if index < 1:
            return signals
        
        if any(x is None for x in [macd[index], macd[index-1], 
                                    signal[index], signal[index-1]]):
            return signals
        
        # Cruzamento bullish (MACD cruza acima da linha de sinal)
        if macd[index-1] <= signal[index-1] and macd[index] > signal[index]:
            signals.append(SignalType.MACD_BULLISH_CROSS)
        
        # Cruzamento bearish (MACD cruza abaixo da linha de sinal)
        if macd[index-1] >= signal[index-1] and macd[index] < signal[index]:
            signals.append(SignalType.MACD_BEARISH_CROSS)
        
        # Reversão do histograma
        if histogram[index] is not None and histogram[index-1] is not None:
            if index >= 2 and histogram[index-2] is not None:
                # Histograma revertendo para cima (de negativo crescendo)
                if (histogram[index-2] < histogram[index-1] < 0 and 
                    histogram[index] > histogram[index-1]):
                    signals.append(SignalType.HISTOGRAM_REVERSAL_UP)
                
                # Histograma revertendo para baixo (de positivo decrescendo)
                if (histogram[index-2] > histogram[index-1] > 0 and 
                    histogram[index] < histogram[index-1]):
                    signals.append(SignalType.HISTOGRAM_REVERSAL_DOWN)
        
        return signals
    
    def detect_divergence(self, prices: List[float], rsi: List[Optional[float]], 
                          index: int) -> Optional[SignalType]:
        """
        Detecta divergências entre preço e RSI.
        
        Divergência Bullish: Preço faz lower low, RSI faz higher low
        Divergência Bearish: Preço faz higher high, RSI faz lower high
        """
        lookback = self.divergence_lookback
        
        if index < lookback:
            return None
        
        # Encontrar mínimos/máximos locais no período
        price_window = prices[index - lookback:index + 1]
        rsi_window = [r for r in rsi[index - lookback:index + 1] if r is not None]
        
        if len(rsi_window) < lookback // 2:
            return None
        
        # Simplificação: comparar primeiro e último valores do período
        price_start = price_window[0]
        price_end = price_window[-1]
        rsi_start = rsi_window[0]
        rsi_end = rsi_window[-1]
        
        # Divergência Bullish: preço caindo, RSI subindo
        if price_end < price_start and rsi_end > rsi_start:
            # Verificar se RSI está em zona de sobrevenda
            if rsi_end < 40:
                return SignalType.BULLISH_DIVERGENCE
        
        # Divergência Bearish: preço subindo, RSI caindo
        if price_end > price_start and rsi_end < rsi_start:
            # Verificar se RSI está em zona de sobrecompra
            if rsi_end > 60:
                return SignalType.BEARISH_DIVERGENCE
        
        return None
    
    def get_momentum_state(self, rsi: float, macd: float, signal: float) -> MomentumState:
        """Determina o estado atual do momentum"""
        
        macd_bullish = macd > signal
        
        if rsi > 70 and macd_bullish:
            return MomentumState.STRONG_BULLISH
        elif rsi > 50 and macd_bullish:
            return MomentumState.BULLISH
        elif rsi < 30 and not macd_bullish:
            return MomentumState.STRONG_BEARISH
        elif rsi < 50 and not macd_bullish:
            return MomentumState.BEARISH
        else:
            return MomentumState.NEUTRAL
    
    def calculate_signal_strength(self, signal_type: SignalType, 
                                   rsi: float, macd: float, signal: float) -> float:
        """Calcula a força do sinal (0-100)"""
        base_strength = 50
        
        # Ajustar baseado no tipo de sinal
        signal_weights = {
            SignalType.RSI_OVERSOLD_EXIT: 70,
            SignalType.RSI_OVERBOUGHT_EXIT: 70,
            SignalType.RSI_CENTERLINE_CROSS_UP: 60,
            SignalType.RSI_CENTERLINE_CROSS_DOWN: 60,
            SignalType.MACD_BULLISH_CROSS: 75,
            SignalType.MACD_BEARISH_CROSS: 75,
            SignalType.BULLISH_DIVERGENCE: 85,
            SignalType.BEARISH_DIVERGENCE: 85,
            SignalType.HISTOGRAM_REVERSAL_UP: 55,
            SignalType.HISTOGRAM_REVERSAL_DOWN: 55,
        }
        
        base_strength = signal_weights.get(signal_type, 50)
        
        # Ajustar baseado na distância do RSI dos extremos
        if rsi < 30:
            base_strength += (30 - rsi) / 2
        elif rsi > 70:
            base_strength += (rsi - 70) / 2
        
        # Ajustar baseado na força do MACD
        macd_diff = abs(macd - signal)
        base_strength += min(10, macd_diff * 100)
        
        return min(100, base_strength)
    
    def analyze(self, candles: List[Dict], index: int = -1) -> List[MomentumSignal]:
        """
        Analisa o momentum no índice especificado e retorna todos os sinais encontrados.
        
        Args:
            candles: Lista de candles
            index: Índice a analisar (-1 para o último)
        
        Returns:
            Lista de MomentumSignals encontrados
        """
        if len(candles) < max(self.macd_slow, self.rsi_period) + 10:
            return []
        
        closes = [c['close'] for c in candles]
        i = index if index >= 0 else len(candles) + index
        
        # Calcular indicadores
        rsi = self.calculate_rsi(closes)
        macd_line, signal_line, histogram = self.calculate_macd(closes)
        
        if any(x is None for x in [rsi[i], macd_line[i], signal_line[i]]):
            return []
        
        signals = []
        
        # Detectar sinais RSI
        rsi_signals = self.detect_rsi_signals(rsi, i)
        for sig_type in rsi_signals:
            direction = 'LONG' if 'UP' in sig_type.value or 'OVERSOLD' in sig_type.value else 'SHORT'
            strength = self.calculate_signal_strength(sig_type, rsi[i], macd_line[i], signal_line[i])
            
            signals.append(MomentumSignal(
                timestamp=candles[i].get('datetime', str(i)),
                signal_type=sig_type,
                direction=direction,
                strength=round(strength, 2),
                rsi_value=round(rsi[i], 2),
                macd_value=round(macd_line[i], 4),
                macd_signal=round(signal_line[i], 4),
                macd_histogram=round(histogram[i], 4) if histogram[i] else 0,
                description=f"RSI Signal: {sig_type.value}"
            ))
        
        # Detectar sinais MACD
        macd_signals = self.detect_macd_signals(macd_line, signal_line, histogram, i)
        for sig_type in macd_signals:
            direction = 'LONG' if 'BULLISH' in sig_type.value or 'UP' in sig_type.value else 'SHORT'
            strength = self.calculate_signal_strength(sig_type, rsi[i], macd_line[i], signal_line[i])
            
            signals.append(MomentumSignal(
                timestamp=candles[i].get('datetime', str(i)),
                signal_type=sig_type,
                direction=direction,
                strength=round(strength, 2),
                rsi_value=round(rsi[i], 2),
                macd_value=round(macd_line[i], 4),
                macd_signal=round(signal_line[i], 4),
                macd_histogram=round(histogram[i], 4) if histogram[i] else 0,
                description=f"MACD Signal: {sig_type.value}"
            ))
        
        # Detectar divergências
        divergence = self.detect_divergence(closes, rsi, i)
        if divergence:
            direction = 'LONG' if 'BULLISH' in divergence.value else 'SHORT'
            strength = self.calculate_signal_strength(divergence, rsi[i], macd_line[i], signal_line[i])
            
            signals.append(MomentumSignal(
                timestamp=candles[i].get('datetime', str(i)),
                signal_type=divergence,
                direction=direction,
                strength=round(strength, 2),
                rsi_value=round(rsi[i], 2),
                macd_value=round(macd_line[i], 4),
                macd_signal=round(signal_line[i], 4),
                macd_histogram=round(histogram[i], 4) if histogram[i] else 0,
                description=f"Divergence: {divergence.value}"
            ))
        
        return signals
    
    def get_current_state(self, candles: List[Dict]) -> Dict:
        """Retorna o estado atual do momentum com todos os valores"""
        if len(candles) < max(self.macd_slow, self.rsi_period) + 5:
            return {'error': 'Insufficient data'}
        
        closes = [c['close'] for c in candles]
        rsi = self.calculate_rsi(closes)
        macd_line, signal_line, histogram = self.calculate_macd(closes)
        
        i = -1
        if any(x is None for x in [rsi[i], macd_line[i], signal_line[i]]):
            return {'error': 'Indicators not ready'}
        
        state = self.get_momentum_state(rsi[i], macd_line[i], signal_line[i])
        
        return {
            'timestamp': candles[-1].get('datetime', 'N/A'),
            'price': closes[-1],
            'rsi': round(rsi[i], 2),
            'macd': round(macd_line[i], 4),
            'macd_signal': round(signal_line[i], 4),
            'macd_histogram': round(histogram[i], 4) if histogram[i] else 0,
            'momentum_state': state.value,
            'rsi_zone': 'oversold' if rsi[i] < 30 else ('overbought' if rsi[i] > 70 else 'neutral'),
            'macd_position': 'above_signal' if macd_line[i] > signal_line[i] else 'below_signal'
        }


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ALGORITMO 3: DETECTOR DE MOMENTUM (RSI + MACD)")
    print("=" * 70)
    
    detector = MomentumDetector(
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9
    )
    
    print("\nParâmetros configurados:")
    print(f"  - RSI: {detector.rsi_period} períodos")
    print(f"  - RSI Sobrevenda: {detector.rsi_oversold}")
    print(f"  - RSI Sobrecompra: {detector.rsi_overbought}")
    print(f"  - MACD: {detector.macd_fast}, {detector.macd_slow}, {detector.macd_signal_period}")
    
    print("\n" + "=" * 70)
    print("SINAIS DETECTADOS:")
    print("=" * 70)
    print("\n[RSI]")
    print("  - RSI_OVERSOLD_EXIT: RSI sai da zona < 30 → LONG")
    print("  - RSI_OVERBOUGHT_EXIT: RSI sai da zona > 70 → SHORT")
    print("  - RSI_CENTERLINE_CROSS_UP: RSI cruza 50 para cima → LONG")
    print("  - RSI_CENTERLINE_CROSS_DOWN: RSI cruza 50 para baixo → SHORT")
    
    print("\n[MACD]")
    print("  - MACD_BULLISH_CROSS: MACD cruza acima da Signal → LONG")
    print("  - MACD_BEARISH_CROSS: MACD cruza abaixo da Signal → SHORT")
    print("  - HISTOGRAM_REVERSAL_UP: Histograma reverte para cima → LONG")
    print("  - HISTOGRAM_REVERSAL_DOWN: Histograma reverte para baixo → SHORT")
    
    print("\n[DIVERGÊNCIAS]")
    print("  - BULLISH_DIVERGENCE: Preço cai, RSI sobe → LONG (forte)")
    print("  - BEARISH_DIVERGENCE: Preço sobe, RSI cai → SHORT (forte)")
