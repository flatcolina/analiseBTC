#!/usr/bin/env python3
"""
=============================================================================
ALGORITMO 5: SISTEMA DE SCORE DINÂMICO E GERENCIAMENTO DE RISCO
=============================================================================

DESCRIÇÃO:
Este algoritmo integra todos os indicadores em um sistema de pontuação 
dinâmico que calcula a qualidade do setup, define automaticamente os níveis
de SL/TP baseados no ATR, e fornece regras de gerenciamento da operação.

LÓGICA DO PADRÃO:
1. Sistema de pontuação (0-100) baseado em múltiplos indicadores
2. Cálculo automático de SL/TP baseado no ATR
3. Regras de gerenciamento dinâmico (breakeven, trailing, saída parcial)
4. Frequência de atualização adaptativa baseada no score

DIFERENCIAL:
- Integra TODOS os indicadores em um único score
- Fornece níveis exatos de entrada, SL e TP
- Inclui regras de gerenciamento pós-entrada
- Adapta a frequência de monitoramento ao contexto

APLICAÇÃO:
- Sistema completo de decisão de trading
- Pode ser usado como base para automação

EXEMPLO DE REFERÊNCIA: Aplicável a todos os 5 exemplos do documento
=============================================================================
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

class SetupQuality(Enum):
    """Qualidade do setup baseada no score"""
    EXCELLENT = "excellent"  # 85-100
    GOOD = "good"           # 70-84
    ACCEPTABLE = "acceptable"  # 55-69
    POOR = "poor"           # 40-54
    AVOID = "avoid"         # 0-39

class TradePhase(Enum):
    """Fases do gerenciamento do trade"""
    ENTRY = "entry"
    INITIAL = "initial"      # 0-50% do alvo
    BREAKEVEN = "breakeven"  # 50-100% do alvo
    TRAILING = "trailing"    # Acima de 100% do alvo
    EXIT = "exit"

@dataclass
class RiskParameters:
    """Parâmetros de risco calculados"""
    entry_price: float
    stop_loss: float
    take_profit_1: float  # TP parcial (1:1)
    take_profit_2: float  # TP final (1:1.5)
    take_profit_3: float  # TP estendido (1:2)
    risk_amount: float    # Distância do SL em $
    reward_1: float       # Distância do TP1 em $
    reward_2: float       # Distância do TP2 em $
    risk_reward_ratio: float
    position_size_suggestion: float  # % do capital sugerido

@dataclass
class TradeManagement:
    """Regras de gerenciamento do trade"""
    move_to_breakeven_at: float  # Preço para mover SL para BE
    partial_exit_at: float       # Preço para saída parcial
    partial_exit_pct: float      # % da posição para sair
    trailing_start_at: float     # Preço para iniciar trailing
    trailing_distance: float     # Distância do trailing em $
    max_hold_candles: int        # Máximo de candles para segurar

@dataclass
class SetupAnalysis:
    """Análise completa do setup"""
    timestamp: str
    direction: str  # 'LONG' ou 'SHORT'
    score: float
    quality: SetupQuality
    risk_params: RiskParameters
    management: TradeManagement
    indicators: Dict
    score_breakdown: Dict
    recommendation: str
    update_frequency_seconds: int  # Frequência de atualização sugerida

class DynamicScoreSystem:
    """
    Sistema de pontuação dinâmico que integra todos os indicadores
    e fornece análise completa com gerenciamento de risco.
    """
    
    def __init__(self,
                 # Parâmetros de indicadores
                 ema_periods: Tuple[int, int, int] = (9, 21, 55),
                 rsi_period: int = 14,
                 macd_params: Tuple[int, int, int] = (12, 26, 9),
                 atr_period: int = 14,
                 # Parâmetros de risco
                 atr_multiplier_sl: float = 1.5,
                 risk_reward_1: float = 1.0,
                 risk_reward_2: float = 1.5,
                 risk_reward_3: float = 2.0,
                 # Parâmetros de gerenciamento
                 breakeven_trigger_pct: float = 0.5,  # 50% do alvo
                 partial_exit_pct: float = 0.5,       # 50% da posição
                 max_risk_per_trade: float = 0.02):   # 2% do capital
        
        self.ema_fast, self.ema_mid, self.ema_slow = ema_periods
        self.rsi_period = rsi_period
        self.macd_fast, self.macd_slow, self.macd_signal = macd_params
        self.atr_period = atr_period
        
        self.atr_multiplier_sl = atr_multiplier_sl
        self.rr1 = risk_reward_1
        self.rr2 = risk_reward_2
        self.rr3 = risk_reward_3
        
        self.breakeven_trigger_pct = breakeven_trigger_pct
        self.partial_exit_pct = partial_exit_pct
        self.max_risk_per_trade = max_risk_per_trade
        
        # Pesos dos indicadores no score
        self.weights = {
            'vwap_alignment': 20,
            'ema_alignment': 25,
            'ema_pullback': 15,
            'rsi_condition': 15,
            'macd_condition': 15,
            'volume_condition': 10
        }
    
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
            return macd_line, [None] * len(macd_line), [None] * len(macd_line)
        
        signal_values = self.calculate_ema(macd_values, self.macd_signal)
        
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
    # SISTEMA DE PONTUAÇÃO
    # =========================================================================
    
    def calculate_score(self, indicators: Dict, direction: str) -> Tuple[float, Dict]:
        """
        Calcula o score total (0-100) baseado em todos os indicadores.
        Retorna o score e o breakdown detalhado.
        """
        breakdown = {}
        total_score = 0
        
        price = indicators['price']
        vwap = indicators['vwap']
        ema9 = indicators['ema9']
        ema21 = indicators['ema21']
        ema55 = indicators['ema55']
        rsi = indicators['rsi']
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        
        # 1. VWAP Alignment (20 pontos)
        if direction == 'LONG':
            if price > vwap:
                vwap_score = self.weights['vwap_alignment']
                # Bônus por distância
                dist_pct = (price - vwap) / vwap * 100
                if 0.1 < dist_pct < 0.5:
                    vwap_score = min(20, vwap_score + 2)
            else:
                vwap_score = 0
        else:  # SHORT
            if price < vwap:
                vwap_score = self.weights['vwap_alignment']
                dist_pct = (vwap - price) / vwap * 100
                if 0.1 < dist_pct < 0.5:
                    vwap_score = min(20, vwap_score + 2)
            else:
                vwap_score = 0
        
        breakdown['vwap_alignment'] = vwap_score
        total_score += vwap_score
        
        # 2. EMA Alignment (25 pontos)
        if direction == 'LONG':
            if ema9 > ema21 > ema55:
                ema_score = self.weights['ema_alignment']
            elif ema9 > ema21:
                ema_score = 15
            else:
                ema_score = 0
        else:  # SHORT
            if ema9 < ema21 < ema55:
                ema_score = self.weights['ema_alignment']
            elif ema9 < ema21:
                ema_score = 15
            else:
                ema_score = 0
        
        breakdown['ema_alignment'] = ema_score
        total_score += ema_score
        
        # 3. EMA Pullback (15 pontos)
        tolerance = 0.002
        dist_ema9 = abs(price - ema9) / ema9
        
        if dist_ema9 <= tolerance:
            pullback_score = self.weights['ema_pullback']
        elif dist_ema9 <= tolerance * 2:
            pullback_score = 10
        elif dist_ema9 <= tolerance * 3:
            pullback_score = 5
        else:
            pullback_score = 0
        
        breakdown['ema_pullback'] = pullback_score
        total_score += pullback_score
        
        # 4. RSI Condition (15 pontos)
        if direction == 'LONG':
            if 50 < rsi < 70:
                rsi_score = self.weights['rsi_condition']
            elif 40 < rsi <= 50:
                rsi_score = 10
            elif rsi <= 30:  # Sobrevenda, bom para reversão
                rsi_score = 12
            else:
                rsi_score = 5
        else:  # SHORT
            if 30 < rsi < 50:
                rsi_score = self.weights['rsi_condition']
            elif 50 <= rsi < 60:
                rsi_score = 10
            elif rsi >= 70:  # Sobrecompra, bom para reversão
                rsi_score = 12
            else:
                rsi_score = 5
        
        breakdown['rsi_condition'] = rsi_score
        total_score += rsi_score
        
        # 5. MACD Condition (15 pontos)
        if direction == 'LONG':
            if macd > macd_signal:
                macd_score = self.weights['macd_condition']
                # Bônus se cruzou recentemente
                if indicators.get('macd_crossed_up', False):
                    macd_score = min(15, macd_score + 3)
            else:
                macd_score = 0
        else:  # SHORT
            if macd < macd_signal:
                macd_score = self.weights['macd_condition']
                if indicators.get('macd_crossed_down', False):
                    macd_score = min(15, macd_score + 3)
            else:
                macd_score = 0
        
        breakdown['macd_condition'] = macd_score
        total_score += macd_score
        
        # 6. Volume Condition (10 pontos) - Simplificado
        volume = indicators.get('volume', 0)
        avg_volume = indicators.get('avg_volume', volume)
        
        if avg_volume > 0 and volume > avg_volume * 1.2:
            volume_score = self.weights['volume_condition']
        elif avg_volume > 0 and volume > avg_volume:
            volume_score = 7
        else:
            volume_score = 5  # Neutro
        
        breakdown['volume_condition'] = volume_score
        total_score += volume_score
        
        return total_score, breakdown
    
    def get_quality(self, score: float) -> SetupQuality:
        """Converte score em qualidade do setup"""
        if score >= 85:
            return SetupQuality.EXCELLENT
        elif score >= 70:
            return SetupQuality.GOOD
        elif score >= 55:
            return SetupQuality.ACCEPTABLE
        elif score >= 40:
            return SetupQuality.POOR
        else:
            return SetupQuality.AVOID
    
    def get_update_frequency(self, score: float, quality: SetupQuality) -> int:
        """
        Determina a frequência de atualização baseada no score.
        Scores mais altos = atualizações mais frequentes.
        """
        frequency_table = {
            SetupQuality.EXCELLENT: 5,    # 5 segundos
            SetupQuality.GOOD: 10,        # 10 segundos
            SetupQuality.ACCEPTABLE: 30,  # 30 segundos
            SetupQuality.POOR: 60,        # 1 minuto
            SetupQuality.AVOID: 120       # 2 minutos
        }
        return frequency_table.get(quality, 60)
    
    # =========================================================================
    # CÁLCULO DE RISCO
    # =========================================================================
    
    def calculate_risk_parameters(self, entry_price: float, atr: float, 
                                   direction: str, capital: float = 10000) -> RiskParameters:
        """Calcula todos os parâmetros de risco"""
        
        risk_amount = atr * self.atr_multiplier_sl
        
        if direction == 'LONG':
            stop_loss = entry_price - risk_amount
            tp1 = entry_price + (risk_amount * self.rr1)
            tp2 = entry_price + (risk_amount * self.rr2)
            tp3 = entry_price + (risk_amount * self.rr3)
        else:  # SHORT
            stop_loss = entry_price + risk_amount
            tp1 = entry_price - (risk_amount * self.rr1)
            tp2 = entry_price - (risk_amount * self.rr2)
            tp3 = entry_price - (risk_amount * self.rr3)
        
        # Calcular tamanho da posição sugerido
        max_loss = capital * self.max_risk_per_trade
        position_size = max_loss / risk_amount
        position_size_pct = (position_size * entry_price) / capital * 100
        
        return RiskParameters(
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(tp1, 2),
            take_profit_2=round(tp2, 2),
            take_profit_3=round(tp3, 2),
            risk_amount=round(risk_amount, 2),
            reward_1=round(risk_amount * self.rr1, 2),
            reward_2=round(risk_amount * self.rr2, 2),
            risk_reward_ratio=self.rr2,
            position_size_suggestion=round(position_size_pct, 2)
        )
    
    def calculate_management_rules(self, risk_params: RiskParameters, 
                                    direction: str, atr: float) -> TradeManagement:
        """Calcula as regras de gerenciamento do trade"""
        
        entry = risk_params.entry_price
        tp1 = risk_params.take_profit_1
        risk = risk_params.risk_amount
        
        if direction == 'LONG':
            # Mover para breakeven quando atingir 50% do TP1
            breakeven_at = entry + (risk * self.breakeven_trigger_pct)
            # Saída parcial no TP1
            partial_at = tp1
            # Trailing começa após TP1
            trailing_start = tp1
        else:  # SHORT
            breakeven_at = entry - (risk * self.breakeven_trigger_pct)
            partial_at = tp1
            trailing_start = tp1
        
        return TradeManagement(
            move_to_breakeven_at=round(breakeven_at, 2),
            partial_exit_at=round(partial_at, 2),
            partial_exit_pct=self.partial_exit_pct,
            trailing_start_at=round(trailing_start, 2),
            trailing_distance=round(atr * 1.0, 2),  # 1x ATR de trailing
            max_hold_candles=60  # Máximo 60 candles (5 horas em 5min)
        )
    
    # =========================================================================
    # ANÁLISE PRINCIPAL
    # =========================================================================
    
    def analyze(self, candles: List[Dict], index: int = -1, 
                capital: float = 10000) -> Optional[SetupAnalysis]:
        """
        Realiza análise completa e retorna setup com score, risco e gerenciamento.
        """
        if len(candles) < max(self.ema_slow, self.macd_slow) + 10:
            return None
        
        closes = [c['close'] for c in candles]
        i = index if index >= 0 else len(candles) + index
        
        # Calcular indicadores
        ema9 = self.calculate_ema(closes, self.ema_fast)
        ema21 = self.calculate_ema(closes, self.ema_mid)
        ema55 = self.calculate_ema(closes, self.ema_slow)
        rsi = self.calculate_rsi(closes)
        macd_line, signal_line, histogram = self.calculate_macd(closes)
        atr = self.calculate_atr(candles)
        vwap = self.calculate_vwap(candles)
        
        if any(x is None for x in [ema9[i], ema21[i], ema55[i], rsi[i], 
                                    macd_line[i], signal_line[i], atr[i], vwap[i]]):
            return None
        
        price = closes[i]
        
        # Determinar direção baseada nas condições
        emas_bullish = ema9[i] > ema21[i] > ema55[i]
        emas_bearish = ema9[i] < ema21[i] < ema55[i]
        above_vwap = price > vwap[i]
        
        if emas_bullish and above_vwap:
            direction = 'LONG'
        elif emas_bearish and not above_vwap:
            direction = 'SHORT'
        else:
            # Sem direção clara, calcular score para ambos e escolher melhor
            direction = 'LONG' if above_vwap else 'SHORT'
        
        # Calcular volume médio
        volumes = [c.get('volume', 0) for c in candles[max(0, i-20):i+1]]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        
        # Construir indicadores
        indicators = {
            'price': price,
            'vwap': vwap[i],
            'ema9': ema9[i],
            'ema21': ema21[i],
            'ema55': ema55[i],
            'rsi': rsi[i],
            'macd': macd_line[i],
            'macd_signal': signal_line[i],
            'macd_histogram': histogram[i] if histogram[i] else 0,
            'atr': atr[i],
            'volume': candles[i].get('volume', 0),
            'avg_volume': avg_volume,
            'macd_crossed_up': (macd_line[i] > signal_line[i] and 
                               macd_line[i-1] <= signal_line[i-1] if i > 0 else False),
            'macd_crossed_down': (macd_line[i] < signal_line[i] and 
                                  macd_line[i-1] >= signal_line[i-1] if i > 0 else False)
        }
        
        # Calcular score
        score, breakdown = self.calculate_score(indicators, direction)
        quality = self.get_quality(score)
        update_freq = self.get_update_frequency(score, quality)
        
        # Calcular parâmetros de risco
        risk_params = self.calculate_risk_parameters(price, atr[i], direction, capital)
        management = self.calculate_management_rules(risk_params, direction, atr[i])
        
        # Gerar recomendação
        if quality == SetupQuality.EXCELLENT:
            recommendation = f"ENTRADA RECOMENDADA: Setup {direction} de alta qualidade (Score: {score:.0f})"
        elif quality == SetupQuality.GOOD:
            recommendation = f"ENTRADA VÁLIDA: Setup {direction} com boa qualidade (Score: {score:.0f})"
        elif quality == SetupQuality.ACCEPTABLE:
            recommendation = f"ENTRADA CAUTELOSA: Setup {direction} aceitável (Score: {score:.0f})"
        elif quality == SetupQuality.POOR:
            recommendation = f"AGUARDAR: Setup {direction} fraco (Score: {score:.0f})"
        else:
            recommendation = f"EVITAR: Condições desfavoráveis (Score: {score:.0f})"
        
        return SetupAnalysis(
            timestamp=candles[i].get('datetime', str(i)),
            direction=direction,
            score=round(score, 2),
            quality=quality,
            risk_params=risk_params,
            management=management,
            indicators={k: round(v, 4) if isinstance(v, float) else v 
                       for k, v in indicators.items()},
            score_breakdown=breakdown,
            recommendation=recommendation,
            update_frequency_seconds=update_freq
        )


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ALGORITMO 5: SISTEMA DE SCORE DINÂMICO E GERENCIAMENTO DE RISCO")
    print("=" * 70)
    
    system = DynamicScoreSystem(
        ema_periods=(9, 21, 55),
        rsi_period=14,
        atr_multiplier_sl=1.5,
        risk_reward_2=1.5,
        max_risk_per_trade=0.02
    )
    
    print("\n" + "=" * 70)
    print("SISTEMA DE PONTUAÇÃO (Total: 100 pontos)")
    print("=" * 70)
    for indicator, weight in system.weights.items():
        print(f"  - {indicator}: {weight} pontos")
    
    print("\n" + "=" * 70)
    print("QUALIDADE DO SETUP:")
    print("=" * 70)
    print("  - EXCELLENT (85-100): Entrada recomendada")
    print("  - GOOD (70-84): Entrada válida")
    print("  - ACCEPTABLE (55-69): Entrada cautelosa")
    print("  - POOR (40-54): Aguardar")
    print("  - AVOID (0-39): Evitar")
    
    print("\n" + "=" * 70)
    print("FREQUÊNCIA DE ATUALIZAÇÃO ADAPTATIVA:")
    print("=" * 70)
    print("  - EXCELLENT: 5 segundos")
    print("  - GOOD: 10 segundos")
    print("  - ACCEPTABLE: 30 segundos")
    print("  - POOR: 60 segundos")
    print("  - AVOID: 120 segundos")
    
    print("\n" + "=" * 70)
    print("REGRAS DE GERENCIAMENTO:")
    print("=" * 70)
    print("  1. Mover SL para Breakeven: Quando atingir 50% do TP1")
    print("  2. Saída Parcial: 50% da posição no TP1 (1:1)")
    print("  3. Trailing Stop: Inicia após TP1, distância de 1x ATR")
    print("  4. TP Final: 1:1.5 RR (restante da posição)")
    print("  5. TP Estendido: 1:2 RR (se momentum continuar)")
