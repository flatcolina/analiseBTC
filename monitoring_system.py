#!/usr/bin/env python3
"""
=============================================================================
SISTEMA DE MONITORAMENTO INTEGRADO
=============================================================================

DESCRIÇÃO:
Este módulo integra todos os 5 algoritmos em um sistema unificado de 
monitoramento que pode ser executado em tempo real para identificar
setups de scalping no BTCUSD.

FUNCIONALIDADES:
1. Integração de todos os detectores
2. Sistema de alertas configurável
3. Frequência de atualização dinâmica baseada no score
4. Logging de sinais e análises
5. Exportação de dados para análise

USO:
- Pode ser executado standalone para monitoramento
- Pode ser importado como módulo em outros sistemas
- Compatível com feeds de dados em tempo real

=============================================================================
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# Importar os algoritmos
from algorithm_1_confluence_long import ConfluenceLongDetector, SetupSignal as LongSignal
from algorithm_2_confluence_short import ConfluenceShortDetector, SetupSignal as ShortSignal
from algorithm_3_momentum_detector import MomentumDetector, MomentumSignal
from algorithm_4_trend_detector import TrendDetector, TrendAnalysis
from algorithm_5_risk_management import DynamicScoreSystem, SetupAnalysis

class AlertLevel(Enum):
    """Níveis de alerta"""
    INFO = "info"
    WARNING = "warning"
    SIGNAL = "signal"
    STRONG_SIGNAL = "strong_signal"

@dataclass
class Alert:
    """Estrutura de alerta"""
    timestamp: str
    level: AlertLevel
    source: str  # Qual algoritmo gerou
    direction: str
    message: str
    score: float
    data: Dict

@dataclass
class MonitoringState:
    """Estado atual do monitoramento"""
    last_update: str
    candles_count: int
    trend_state: str
    momentum_state: str
    vwap_bias: str
    current_score: float
    active_signals: List[Dict]
    update_frequency: int

class ScalpingMonitor:
    """
    Sistema de monitoramento integrado para scalping.
    Combina todos os algoritmos e fornece uma interface unificada.
    """
    
    def __init__(self,
                 # Configurações gerais
                 min_score_for_alert: float = 55,
                 strong_signal_score: float = 75,
                 # Callbacks
                 on_alert: Optional[Callable[[Alert], None]] = None,
                 on_signal: Optional[Callable[[Dict], None]] = None):
        
        self.min_score = min_score_for_alert
        self.strong_score = strong_signal_score
        self.on_alert = on_alert
        self.on_signal = on_signal
        
        # Inicializar todos os detectores
        self.long_detector = ConfluenceLongDetector()
        self.short_detector = ConfluenceShortDetector()
        self.momentum_detector = MomentumDetector()
        self.trend_detector = TrendDetector()
        self.score_system = DynamicScoreSystem()
        
        # Estado interno
        self.alerts_history: List[Alert] = []
        self.signals_history: List[Dict] = []
        self.last_state: Optional[MonitoringState] = None
        
        # Configuração de frequência dinâmica
        self.base_frequency = 30  # segundos
        self.current_frequency = self.base_frequency
    
    def _create_alert(self, level: AlertLevel, source: str, direction: str,
                      message: str, score: float, data: Dict) -> Alert:
        """Cria um novo alerta"""
        alert = Alert(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            level=level,
            source=source,
            direction=direction,
            message=message,
            score=score,
            data=data
        )
        
        self.alerts_history.append(alert)
        
        if self.on_alert:
            self.on_alert(alert)
        
        return alert
    
    def _update_frequency(self, score: float):
        """Atualiza a frequência de monitoramento baseada no score"""
        if score >= 85:
            self.current_frequency = 5
        elif score >= 70:
            self.current_frequency = 10
        elif score >= 55:
            self.current_frequency = 30
        elif score >= 40:
            self.current_frequency = 60
        else:
            self.current_frequency = 120
    
    def analyze(self, candles: List[Dict], capital: float = 10000) -> Dict:
        """
        Realiza análise completa usando todos os algoritmos.
        
        Args:
            candles: Lista de candles OHLCV
            capital: Capital disponível para cálculo de posição
        
        Returns:
            Dicionário com análise completa e sinais
        """
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'candles_analyzed': len(candles),
            'trend_analysis': None,
            'momentum_signals': [],
            'long_setup': None,
            'short_setup': None,
            'score_analysis': None,
            'alerts': [],
            'recommendation': None,
            'update_frequency': self.current_frequency
        }
        
        if len(candles) < 60:
            result['error'] = 'Insufficient data (need at least 60 candles)'
            return result
        
        # 1. Análise de Tendência (Algoritmo 4)
        trend = self.trend_detector.analyze(candles)
        if trend:
            result['trend_analysis'] = {
                'state': trend.trend_state.value,
                'strength': trend.trend_strength,
                'vwap_bias': trend.vwap_bias,
                'ema_alignment': trend.ema_alignment,
                'pullback_zone': trend.pullback_zone.value,
                'entry_quality': trend.entry_quality,
                'recommendation': trend.recommendation
            }
        
        # 2. Análise de Momentum (Algoritmo 3)
        momentum_signals = self.momentum_detector.analyze(candles)
        for sig in momentum_signals:
            result['momentum_signals'].append({
                'type': sig.signal_type.value,
                'direction': sig.direction,
                'strength': sig.strength,
                'rsi': sig.rsi_value,
                'macd': sig.macd_value,
                'description': sig.description
            })
        
        # 3. Detecção de Setup LONG (Algoritmo 1)
        long_setup = self.long_detector.detect_setup(candles)
        if long_setup:
            result['long_setup'] = {
                'entry': long_setup.entry_price,
                'stop_loss': long_setup.stop_loss,
                'take_profit': long_setup.take_profit,
                'strength': long_setup.strength.value,
                'score': long_setup.score,
                'reason': long_setup.reason
            }
            
            if long_setup.score >= self.strong_score:
                self._create_alert(
                    AlertLevel.STRONG_SIGNAL,
                    'ConfluenceLong',
                    'LONG',
                    f"Setup LONG forte detectado! Score: {long_setup.score}",
                    long_setup.score,
                    result['long_setup']
                )
            elif long_setup.score >= self.min_score:
                self._create_alert(
                    AlertLevel.SIGNAL,
                    'ConfluenceLong',
                    'LONG',
                    f"Setup LONG detectado. Score: {long_setup.score}",
                    long_setup.score,
                    result['long_setup']
                )
        
        # 4. Detecção de Setup SHORT (Algoritmo 2)
        short_setup = self.short_detector.detect_setup(candles)
        if short_setup:
            result['short_setup'] = {
                'entry': short_setup.entry_price,
                'stop_loss': short_setup.stop_loss,
                'take_profit': short_setup.take_profit,
                'strength': short_setup.strength.value,
                'score': short_setup.score,
                'reason': short_setup.reason
            }
            
            if short_setup.score >= self.strong_score:
                self._create_alert(
                    AlertLevel.STRONG_SIGNAL,
                    'ConfluenceShort',
                    'SHORT',
                    f"Setup SHORT forte detectado! Score: {short_setup.score}",
                    short_setup.score,
                    result['short_setup']
                )
            elif short_setup.score >= self.min_score:
                self._create_alert(
                    AlertLevel.SIGNAL,
                    'ConfluenceShort',
                    'SHORT',
                    f"Setup SHORT detectado. Score: {short_setup.score}",
                    short_setup.score,
                    result['short_setup']
                )
        
        # 5. Análise de Score Completa (Algoritmo 5)
        score_analysis = self.score_system.analyze(candles, capital=capital)
        if score_analysis:
            result['score_analysis'] = {
                'direction': score_analysis.direction,
                'score': score_analysis.score,
                'quality': score_analysis.quality.value,
                'breakdown': score_analysis.score_breakdown,
                'risk_params': {
                    'entry': score_analysis.risk_params.entry_price,
                    'stop_loss': score_analysis.risk_params.stop_loss,
                    'tp1': score_analysis.risk_params.take_profit_1,
                    'tp2': score_analysis.risk_params.take_profit_2,
                    'tp3': score_analysis.risk_params.take_profit_3,
                    'risk_amount': score_analysis.risk_params.risk_amount,
                    'position_size_pct': score_analysis.risk_params.position_size_suggestion
                },
                'management': {
                    'breakeven_at': score_analysis.management.move_to_breakeven_at,
                    'partial_exit_at': score_analysis.management.partial_exit_at,
                    'partial_exit_pct': score_analysis.management.partial_exit_pct,
                    'trailing_start': score_analysis.management.trailing_start_at,
                    'trailing_distance': score_analysis.management.trailing_distance
                },
                'recommendation': score_analysis.recommendation
            }
            
            # Atualizar frequência baseada no score
            self._update_frequency(score_analysis.score)
            result['update_frequency'] = self.current_frequency
        
        # 6. Gerar recomendação final
        result['recommendation'] = self._generate_final_recommendation(result)
        
        # 7. Adicionar alertas gerados nesta análise
        result['alerts'] = [asdict(a) for a in self.alerts_history[-10:]]
        
        return result
    
    def _generate_final_recommendation(self, analysis: Dict) -> Dict:
        """Gera recomendação final baseada em todas as análises"""
        
        recommendation = {
            'action': 'WAIT',
            'direction': None,
            'confidence': 0,
            'reasons': []
        }
        
        # Verificar se há setup de confluência
        long_score = analysis['long_setup']['score'] if analysis['long_setup'] else 0
        short_score = analysis['short_setup']['score'] if analysis['short_setup'] else 0
        
        # Verificar score geral
        general_score = analysis['score_analysis']['score'] if analysis['score_analysis'] else 0
        
        # Verificar tendência
        trend_favorable = False
        if analysis['trend_analysis']:
            trend_state = analysis['trend_analysis']['state']
            if 'uptrend' in trend_state and long_score > 0:
                trend_favorable = True
                recommendation['reasons'].append(f"Tendência de alta: {trend_state}")
            elif 'downtrend' in trend_state and short_score > 0:
                trend_favorable = True
                recommendation['reasons'].append(f"Tendência de baixa: {trend_state}")
        
        # Verificar momentum
        momentum_favorable = False
        for sig in analysis['momentum_signals']:
            if sig['strength'] > 70:
                momentum_favorable = True
                recommendation['reasons'].append(f"Momentum forte: {sig['description']}")
        
        # Determinar ação
        if long_score >= 75 and trend_favorable:
            recommendation['action'] = 'LONG'
            recommendation['direction'] = 'LONG'
            recommendation['confidence'] = min(100, (long_score + general_score) / 2)
            recommendation['reasons'].append(f"Setup LONG com score {long_score}")
        elif short_score >= 75 and trend_favorable:
            recommendation['action'] = 'SHORT'
            recommendation['direction'] = 'SHORT'
            recommendation['confidence'] = min(100, (short_score + general_score) / 2)
            recommendation['reasons'].append(f"Setup SHORT com score {short_score}")
        elif long_score >= 55 or short_score >= 55:
            recommendation['action'] = 'PREPARE'
            recommendation['direction'] = 'LONG' if long_score > short_score else 'SHORT'
            recommendation['confidence'] = max(long_score, short_score)
            recommendation['reasons'].append("Setup em formação, aguardar confirmação")
        else:
            recommendation['action'] = 'WAIT'
            recommendation['reasons'].append("Sem setup válido no momento")
        
        return recommendation
    
    def get_state(self) -> Dict:
        """Retorna o estado atual do monitoramento"""
        return {
            'alerts_count': len(self.alerts_history),
            'signals_count': len(self.signals_history),
            'current_frequency': self.current_frequency,
            'last_state': asdict(self.last_state) if self.last_state else None
        }
    
    def export_history(self, filepath: str):
        """Exporta histórico de alertas e sinais para arquivo JSON"""
        data = {
            'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'alerts': [asdict(a) for a in self.alerts_history],
            'signals': self.signals_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

def example_alert_handler(alert: Alert):
    """Handler de exemplo para alertas"""
    level_emoji = {
        AlertLevel.INFO: "ℹ️",
        AlertLevel.WARNING: "⚠️",
        AlertLevel.SIGNAL: "🔔",
        AlertLevel.STRONG_SIGNAL: "🚨"
    }
    
    print(f"\n{level_emoji.get(alert.level, '•')} [{alert.level.value.upper()}] {alert.timestamp}")
    print(f"   Fonte: {alert.source}")
    print(f"   Direção: {alert.direction}")
    print(f"   Score: {alert.score}")
    print(f"   Mensagem: {alert.message}")

if __name__ == "__main__":
    print("=" * 70)
    print("SISTEMA DE MONITORAMENTO INTEGRADO")
    print("=" * 70)
    
    # Criar monitor com handler de alertas
    monitor = ScalpingMonitor(
        min_score_for_alert=55,
        strong_signal_score=75,
        on_alert=example_alert_handler
    )
    
    print("\nSistema inicializado com os seguintes detectores:")
    print("  1. ConfluenceLongDetector - Setups de compra")
    print("  2. ConfluenceShortDetector - Setups de venda")
    print("  3. MomentumDetector - Análise de momentum")
    print("  4. TrendDetector - Análise de tendência")
    print("  5. DynamicScoreSystem - Score e gerenciamento de risco")
    
    print("\n" + "=" * 70)
    print("COMO USAR:")
    print("=" * 70)
    print("""
    # Importar o monitor
    from monitoring_system import ScalpingMonitor
    
    # Criar instância
    monitor = ScalpingMonitor(
        min_score_for_alert=55,
        strong_signal_score=75,
        on_alert=my_alert_handler  # Função callback opcional
    )
    
    # Analisar candles
    result = monitor.analyze(candles, capital=10000)
    
    # Verificar recomendação
    print(result['recommendation'])
    
    # Verificar setups específicos
    if result['long_setup']:
        print(f"LONG: Entry={result['long_setup']['entry']}")
    
    # Exportar histórico
    monitor.export_history('signals_history.json')
    """)
    
    print("\n" + "=" * 70)
    print("FREQUÊNCIA DE ATUALIZAÇÃO DINÂMICA:")
    print("=" * 70)
    print("  Score >= 85: Atualizar a cada 5 segundos")
    print("  Score >= 70: Atualizar a cada 10 segundos")
    print("  Score >= 55: Atualizar a cada 30 segundos")
    print("  Score >= 40: Atualizar a cada 60 segundos")
    print("  Score < 40:  Atualizar a cada 120 segundos")
