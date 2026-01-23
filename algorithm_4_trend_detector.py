#!/usr/bin/env python3
"""
=============================================================================
ALGORITMO 4: DETECTOR DE TENDÊNCIA (EMAs + VWAP)
=============================================================================

DESCRIÇÃO:
Este algoritmo foca na identificação e confirmação de tendências utilizando
múltiplas EMAs e o VWAP como referência de valor justo intraday. É ideal
para identificar o contexto macro antes de buscar entradas de scalping.

LÓGICA DO PADRÃO:
1. Alinhamento das EMAs (9, 21, 55) para determinar tendência
2. Posição do preço em relação à VWAP para viés intraday
3. Distância entre EMAs para medir força da tendência
4. Cruzamentos de EMAs para identificar mudanças de tendência
5. Pullbacks para EMAs como zonas de entrada

DIFERENCIAL:
- Foco na estrutura de tendência, não apenas momentum
- Identifica zonas de pullback ideais
- Mede a "saúde" da tendência atual

APLICAÇÃO:
- Timeframe: 5 a 15 minutos para contexto, 1 minuto para entrada
- Ideal para confirmar direção antes de operar

EXEMPLO DE REFERÊNCIA: Base de todos os 5 exemplos do documento
=============================================================================
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class TrendState(Enum):
    """Estados possíveis da tendência"""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

class PullbackZone(Enum):
    """Zonas de pullback identificadas"""
    EMA9 = "ema9"
    EMA21 = "ema21"
    EMA55 = "ema55"
    VWAP = "vwap"
    NONE = "none"

@dataclass
class TrendAnalysis:
    """Resultado da análise de tendência"""
    timestamp: str
    trend_state: TrendState
    trend_strength: float  # 0-100
    vwap_bias: str  # 'bullish', 'bearish', 'neutral'
    ema_alignment: str  # 'bullish', 'bearish', 'mixed'
    pullback_zone: PullbackZone
    entry_quality: float  # 0-100, qualidade da zona de entrada
    price: float
    vwap: float
    ema9: float
    ema21: float
    ema55: float
    ema_spread: float  # Distância percentual entre EMAs
    recommendation: str

class TrendDetector:
    """
    Detector de tendência baseado em EMAs e VWAP.
    Identifica a direção, força e zonas de entrada em tendências.
    """
    
    def __init__(self,
                 ema_periods: Tuple[int, int, int] = (9, 21, 55),
                 pullback_tolerance: float = 0.003,  # 0.3% de tolerância
                 strong_trend_spread: float = 0.005):  # 0.5% spread entre EMAs
        
        self.ema_fast, self.ema_mid, self.ema_slow = ema_periods
        self.pullback_tolerance = pullback_tolerance
        self.strong_trend_spread = strong_trend_spread
    
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
    # ANÁLISE DE TENDÊNCIA
    # =========================================================================
    
    def get_ema_alignment(self, ema9: float, ema21: float, ema55: float) -> str:
        """Determina o alinhamento das EMAs"""
        if ema9 > ema21 > ema55:
            return 'bullish'
        elif ema9 < ema21 < ema55:
            return 'bearish'
        else:
            return 'mixed'
    
    def get_vwap_bias(self, price: float, vwap: float, tolerance: float = 0.001) -> str:
        """Determina o viés baseado na VWAP"""
        diff_pct = (price - vwap) / vwap
        
        if diff_pct > tolerance:
            return 'bullish'
        elif diff_pct < -tolerance:
            return 'bearish'
        else:
            return 'neutral'
    
    def calculate_ema_spread(self, ema9: float, ema21: float, ema55: float) -> float:
        """
        Calcula o spread percentual entre as EMAs.
        Maior spread = tendência mais forte.
        """
        # Spread entre EMA9 e EMA55 como percentual
        spread = abs(ema9 - ema55) / ema55
        return spread
    
    def identify_pullback_zone(self, price: float, ema9: float, ema21: float, 
                                ema55: float, vwap: float, 
                                trend_direction: str) -> Tuple[PullbackZone, float]:
        """
        Identifica em qual zona de pullback o preço está.
        Retorna a zona e a qualidade da entrada (0-100).
        """
        tolerance = self.pullback_tolerance
        
        # Calcular distâncias percentuais
        dist_ema9 = abs(price - ema9) / ema9
        dist_ema21 = abs(price - ema21) / ema21
        dist_ema55 = abs(price - ema55) / ema55
        dist_vwap = abs(price - vwap) / vwap
        
        # Em tendência de alta, pullbacks ideais são para EMAs por baixo
        if trend_direction == 'bullish':
            # Melhor: preço próximo à EMA9, acima das outras
            if dist_ema9 <= tolerance and price >= ema9 * 0.998:
                quality = 90 - (dist_ema9 * 1000)  # Quanto mais próximo, melhor
                return PullbackZone.EMA9, max(0, min(100, quality))
            
            # Bom: preço entre EMA9 e EMA21
            if ema21 <= price <= ema9:
                quality = 75 - (dist_ema21 * 500)
                return PullbackZone.EMA21, max(0, min(100, quality))
            
            # Aceitável: preço próximo à EMA21
            if dist_ema21 <= tolerance * 1.5:
                quality = 60 - (dist_ema21 * 500)
                return PullbackZone.EMA21, max(0, min(100, quality))
        
        # Em tendência de baixa, pullbacks ideais são para EMAs por cima
        elif trend_direction == 'bearish':
            if dist_ema9 <= tolerance and price <= ema9 * 1.002:
                quality = 90 - (dist_ema9 * 1000)
                return PullbackZone.EMA9, max(0, min(100, quality))
            
            if ema9 <= price <= ema21:
                quality = 75 - (dist_ema21 * 500)
                return PullbackZone.EMA21, max(0, min(100, quality))
            
            if dist_ema21 <= tolerance * 1.5:
                quality = 60 - (dist_ema21 * 500)
                return PullbackZone.EMA21, max(0, min(100, quality))
        
        return PullbackZone.NONE, 0
    
    def determine_trend_state(self, ema_alignment: str, vwap_bias: str, 
                               ema_spread: float, price: float, 
                               ema9: float, ema55: float) -> Tuple[TrendState, float]:
        """
        Determina o estado da tendência e sua força.
        """
        strength = 50  # Base
        
        # Tendência de alta
        if ema_alignment == 'bullish':
            if vwap_bias == 'bullish' and ema_spread > self.strong_trend_spread:
                state = TrendState.STRONG_UPTREND
                strength = 85 + min(15, ema_spread * 1000)
            elif vwap_bias == 'bullish':
                state = TrendState.UPTREND
                strength = 70 + min(15, ema_spread * 500)
            else:
                state = TrendState.WEAK_UPTREND
                strength = 55 + min(10, ema_spread * 300)
        
        # Tendência de baixa
        elif ema_alignment == 'bearish':
            if vwap_bias == 'bearish' and ema_spread > self.strong_trend_spread:
                state = TrendState.STRONG_DOWNTREND
                strength = 85 + min(15, ema_spread * 1000)
            elif vwap_bias == 'bearish':
                state = TrendState.DOWNTREND
                strength = 70 + min(15, ema_spread * 500)
            else:
                state = TrendState.WEAK_DOWNTREND
                strength = 55 + min(10, ema_spread * 300)
        
        # Sem tendência clara
        else:
            state = TrendState.RANGING
            strength = 30 + min(20, (1 - ema_spread) * 100)
        
        return state, min(100, strength)
    
    def generate_recommendation(self, trend_state: TrendState, 
                                 pullback_zone: PullbackZone,
                                 entry_quality: float) -> str:
        """Gera uma recomendação baseada na análise"""
        
        if trend_state in [TrendState.STRONG_UPTREND, TrendState.UPTREND]:
            if pullback_zone != PullbackZone.NONE and entry_quality > 60:
                return f"LONG: Tendência de alta confirmada. Pullback para {pullback_zone.value} com qualidade {entry_quality:.0f}%"
            elif pullback_zone != PullbackZone.NONE:
                return f"AGUARDAR: Pullback identificado mas qualidade baixa ({entry_quality:.0f}%)"
            else:
                return "AGUARDAR: Tendência de alta, mas sem pullback para entrada"
        
        elif trend_state in [TrendState.STRONG_DOWNTREND, TrendState.DOWNTREND]:
            if pullback_zone != PullbackZone.NONE and entry_quality > 60:
                return f"SHORT: Tendência de baixa confirmada. Pullback para {pullback_zone.value} com qualidade {entry_quality:.0f}%"
            elif pullback_zone != PullbackZone.NONE:
                return f"AGUARDAR: Pullback identificado mas qualidade baixa ({entry_quality:.0f}%)"
            else:
                return "AGUARDAR: Tendência de baixa, mas sem pullback para entrada"
        
        elif trend_state == TrendState.RANGING:
            return "EVITAR: Mercado em consolidação, sem tendência clara"
        
        else:
            return "CAUTELA: Tendência fraca, aguardar confirmação"
    
    def analyze(self, candles: List[Dict], index: int = -1) -> Optional[TrendAnalysis]:
        """
        Realiza análise completa de tendência no índice especificado.
        
        Args:
            candles: Lista de candles
            index: Índice a analisar (-1 para o último)
        
        Returns:
            TrendAnalysis com todos os dados da análise
        """
        if len(candles) < self.ema_slow + 10:
            return None
        
        closes = [c['close'] for c in candles]
        i = index if index >= 0 else len(candles) + index
        
        # Calcular indicadores
        ema9 = self.calculate_ema(closes, self.ema_fast)
        ema21 = self.calculate_ema(closes, self.ema_mid)
        ema55 = self.calculate_ema(closes, self.ema_slow)
        vwap = self.calculate_vwap(candles)
        
        if any(x is None for x in [ema9[i], ema21[i], ema55[i], vwap[i]]):
            return None
        
        price = closes[i]
        
        # Análises
        ema_alignment = self.get_ema_alignment(ema9[i], ema21[i], ema55[i])
        vwap_bias = self.get_vwap_bias(price, vwap[i])
        ema_spread = self.calculate_ema_spread(ema9[i], ema21[i], ema55[i])
        
        trend_state, trend_strength = self.determine_trend_state(
            ema_alignment, vwap_bias, ema_spread, price, ema9[i], ema55[i]
        )
        
        # Determinar direção para pullback
        trend_direction = 'bullish' if 'UPTREND' in trend_state.value else (
            'bearish' if 'DOWNTREND' in trend_state.value else 'neutral'
        )
        
        pullback_zone, entry_quality = self.identify_pullback_zone(
            price, ema9[i], ema21[i], ema55[i], vwap[i], trend_direction
        )
        
        recommendation = self.generate_recommendation(trend_state, pullback_zone, entry_quality)
        
        return TrendAnalysis(
            timestamp=candles[i].get('datetime', str(i)),
            trend_state=trend_state,
            trend_strength=round(trend_strength, 2),
            vwap_bias=vwap_bias,
            ema_alignment=ema_alignment,
            pullback_zone=pullback_zone,
            entry_quality=round(entry_quality, 2),
            price=round(price, 2),
            vwap=round(vwap[i], 2),
            ema9=round(ema9[i], 2),
            ema21=round(ema21[i], 2),
            ema55=round(ema55[i], 2),
            ema_spread=round(ema_spread * 100, 4),  # Em percentual
            recommendation=recommendation
        )
    
    def detect_ema_crossovers(self, candles: List[Dict], lookback: int = 5) -> List[Dict]:
        """
        Detecta cruzamentos de EMAs nos últimos N candles.
        Útil para identificar mudanças de tendência.
        """
        if len(candles) < self.ema_slow + lookback:
            return []
        
        closes = [c['close'] for c in candles]
        ema9 = self.calculate_ema(closes, self.ema_fast)
        ema21 = self.calculate_ema(closes, self.ema_mid)
        ema55 = self.calculate_ema(closes, self.ema_slow)
        
        crossovers = []
        start_idx = len(candles) - lookback
        
        for i in range(start_idx, len(candles)):
            if i < 1 or any(x is None for x in [ema9[i], ema9[i-1], ema21[i], ema21[i-1]]):
                continue
            
            # EMA9 cruza EMA21 para cima (bullish)
            if ema9[i-1] <= ema21[i-1] and ema9[i] > ema21[i]:
                crossovers.append({
                    'timestamp': candles[i].get('datetime', str(i)),
                    'type': 'EMA9_CROSS_EMA21_UP',
                    'direction': 'BULLISH',
                    'significance': 'HIGH'
                })
            
            # EMA9 cruza EMA21 para baixo (bearish)
            if ema9[i-1] >= ema21[i-1] and ema9[i] < ema21[i]:
                crossovers.append({
                    'timestamp': candles[i].get('datetime', str(i)),
                    'type': 'EMA9_CROSS_EMA21_DOWN',
                    'direction': 'BEARISH',
                    'significance': 'HIGH'
                })
            
            # EMA21 cruza EMA55 (mais significativo)
            if ema21[i] is not None and ema55[i] is not None:
                if ema21[i-1] <= ema55[i-1] and ema21[i] > ema55[i]:
                    crossovers.append({
                        'timestamp': candles[i].get('datetime', str(i)),
                        'type': 'EMA21_CROSS_EMA55_UP',
                        'direction': 'BULLISH',
                        'significance': 'VERY_HIGH'
                    })
                
                if ema21[i-1] >= ema55[i-1] and ema21[i] < ema55[i]:
                    crossovers.append({
                        'timestamp': candles[i].get('datetime', str(i)),
                        'type': 'EMA21_CROSS_EMA55_DOWN',
                        'direction': 'BEARISH',
                        'significance': 'VERY_HIGH'
                    })
        
        return crossovers


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ALGORITMO 4: DETECTOR DE TENDÊNCIA (EMAs + VWAP)")
    print("=" * 70)
    
    detector = TrendDetector(
        ema_periods=(9, 21, 55),
        pullback_tolerance=0.003,
        strong_trend_spread=0.005
    )
    
    print("\nParâmetros configurados:")
    print(f"  - EMAs: {detector.ema_fast}, {detector.ema_mid}, {detector.ema_slow}")
    print(f"  - Tolerância de Pullback: {detector.pullback_tolerance * 100}%")
    print(f"  - Spread para Tendência Forte: {detector.strong_trend_spread * 100}%")
    
    print("\n" + "=" * 70)
    print("ESTADOS DE TENDÊNCIA:")
    print("=" * 70)
    for state in TrendState:
        print(f"  - {state.value}")
    
    print("\n" + "=" * 70)
    print("ZONAS DE PULLBACK:")
    print("=" * 70)
    print("  - EMA9: Melhor zona (90% qualidade máxima)")
    print("  - EMA21: Boa zona (75% qualidade máxima)")
    print("  - EMA55: Zona de suporte/resistência forte")
    print("  - VWAP: Referência de valor justo intraday")
    
    print("\n" + "=" * 70)
    print("LÓGICA DE ANÁLISE:")
    print("=" * 70)
    print("1. Verificar alinhamento das EMAs (bullish/bearish/mixed)")
    print("2. Verificar posição do preço vs VWAP (viés intraday)")
    print("3. Calcular spread entre EMAs (força da tendência)")
    print("4. Identificar zona de pullback atual")
    print("5. Calcular qualidade da entrada (0-100)")
    print("6. Gerar recomendação baseada na análise completa")
