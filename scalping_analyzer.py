# -*- coding: utf-8 -*-
"""
Módulo de Análise de Scalping Integrada.
Integra os 5 algoritmos de análise para gerar um relatório unificado.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict as dataclass_asdict

# Importar modelos de dados e classes dos algoritmos
from models import Candle, FullAnalysis, SetupAnalysis, MomentumSignal, TrendAnalysis, RiskParams, ManagementParams, ScoreAnalysis
from algorithm_1_confluence_long import ConfluenceLongDetector
from algorithm_2_confluence_short import ConfluenceShortDetector
from algorithm_3_momentum_detector import MomentumDetector
from algorithm_4_trend_detector import TrendDetector
from algorithm_5_risk_management import DynamicScoreSystem

def _convert_candles_to_dict(candles: List[Candle]) -> List[Dict[str, Any]]:
    """Converte a lista de objetos Candle (dataclass) para a lista de dicionários esperada pelos algoritmos."""
    converted = []
    for c in candles:
        # Usar asdict para converter a dataclass Candle em dict
        c_dict = dataclass_asdict(c)
        # Adicionar o campo 'datetime' que os algoritmos esperam
        c_dict['datetime'] = datetime.fromtimestamp(c.close_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
        converted.append(c_dict)
    return converted

class ScalpingAnalyzer:
    """
    Classe principal para executar a análise de scalping integrada.
    """
    def __init__(self):
        self.long_detector = ConfluenceLongDetector()
        self.short_detector = ConfluenceShortDetector()
        self.momentum_detector = MomentumDetector()
        self.trend_detector = TrendDetector()
        self.score_system = DynamicScoreSystem()

    def analyze_all(self, candles: List[Candle], capital: float = 10000) -> Dict[str, Any]:
        """
        Executa todos os 5 algoritmos e retorna um dicionário de resultados.
        """
        if len(candles) < 60:
            return {"error": "Insufficient data (need at least 60 candles)"}

        # 1. Converter candles para o formato esperado pelos algoritmos
        candles_dict = _convert_candles_to_dict(candles)

        # 2. Executar os detectores de setup (Algoritmos 1 e 2)
        long_setup = self.long_detector.detect_setup(candles_dict)
        short_setup = self.short_detector.detect_setup(candles_dict)

        # 3. Executar análise de Momentum (Algoritmo 3)
        momentum_signals = self.momentum_detector.analyze(candles_dict)

        # 4. Executar análise de Tendência (Algoritmo 4)
        trend_analysis = self.trend_detector.analyze(candles_dict)

        # 5. Executar Sistema de Score Dinâmico (Algoritmo 5)
        score_analysis = self.score_system.analyze(candles_dict, capital=capital)

        # 6. Compilar resultados
        result = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "candles_analyzed": len(candles),
            "long_setup": asdict(long_setup) if long_setup else None,
            "short_setup": asdict(short_setup) if short_setup else None,
            "momentum_analysis": [asdict(s) for s in momentum_signals],
            "trend_analysis": asdict(trend_analysis) if trend_analysis else None,
            "score_analysis": asdict(score_analysis) if score_analysis else None,
        }

            # Limpar enum values para strings no score_analysis
            # Isso não é mais necessário se os algoritmos usarem as dataclasses de models.py
            # e o _safe_asdict for usado corretamente.

            # O problema é que os algoritmos originais usam classes internas com Enums.
            # Vamos manter o _safe_asdict e o _clean_dataclass_dict, mas simplificar a classe principal.

            # 6. Compilar resultados
            # Usar a dataclass FullAnalysis para garantir a estrutura de saída
            full_analysis = FullAnalysis(
                snapshot=candles[-1], # A última vela já tem todos os indicadores
                long_setup=_safe_asdict(long_setup),
                short_setup=_safe_asdict(short_setup),
                momentum_analysis=[_safe_asdict(s) for s in momentum_signals],
                trend_analysis=_safe_asdict(trend_analysis),
                score_analysis=_safe_asdict(score_analysis),
            )

            # Retornar o dicionário da dataclass FullAnalysis
            return dataclass_asdict(full_analysis)

    # Helper para limpar Enums e converter dataclasses para dict
    def _clean_dataclass_dict(data: Dict) -> Dict:
        """Limpa os dicionários de dataclasses, convertendo Enums para strings."""
        if isinstance(data, dict):
            return {k: _clean_dataclass_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [_clean_dataclass_dict(v) for v in data]
        elif hasattr(data, 'value'):
            return data.value
        return data

    def _safe_asdict(obj):
        """Tenta usar dataclasses.asdict e limpa Enums. Usado para os resultados dos algoritmos."""
        try:
            data = dataclass_asdict(obj)
            return _clean_dataclass_dict(data)
        except TypeError:
            # Se não for um dataclass, retorna o objeto
            return obj

    class ScalpingAnalyzer:
        """
        Classe principal para executar a análise de scalping integrada.
        """
        def __init__(self):
            # Instanciar os detectores dos algoritmos originais
            self.long_detector = ConfluenceLongDetector()
            self.short_detector = ConfluenceShortDetector()
            self.momentum_detector = MomentumDetector()
            self.trend_detector = TrendDetector()
            self.score_system = DynamicScoreSystem()

        def get_full_analysis(self, candles: List[Candle], capital: float = 10000) -> Dict[str, Any]:
            """
            Executa todos os 5 algoritmos e retorna um dicionário de resultados.
            """
            if len(candles) < 60:
                return {"error": "Insufficient data (need at least 60 candles)"}

            candles_dict = _convert_candles_to_dict(candles)

            # 1 & 2. Confluência
            long_setup = self.long_detector.detect_setup(candles_dict)
            short_setup = self.short_detector.detect_setup(candles_dict)

            # 3. Momentum
            momentum_signals = self.momentum_detector.analyze(candles_dict)

            # 4. Tendência
            trend_analysis = self.trend_detector.analyze(candles_dict)

            # 5. Score e Risco
            score_analysis = self.score_system.analyze(candles_dict, capital=capital)

            # 6. Compilar resultados na estrutura FullAnalysis
            full_analysis = FullAnalysis(
                snapshot=candles[-1], # A última vela já tem todos os indicadores
                long_setup=_safe_asdict(long_setup),
                short_setup=_safe_asdict(short_setup),
                momentum_analysis=[_safe_asdict(s) for s in momentum_signals],
                trend_analysis=_safe_asdict(trend_analysis),
                score_analysis=_safe_asdict(score_analysis),
            )

            # Retornar o dicionário da dataclass FullAnalysis
            return dataclass_asdict(full_analysis)

    # Instanciar o analisador para uso no main.py
    analyzer = ScalpingAnalyzer()
