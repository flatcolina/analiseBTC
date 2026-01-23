# -*- coding: utf-8 -*-
"""
Módulo de Análise de Scalping Integrada.
Integra os 5 algoritmos de análise para gerar um relatório unificado.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime

# Importar os 5 algoritmos
from algorithm_1_confluence_long import ConfluenceLongDetector
from algorithm_2_confluence_short import ConfluenceShortDetector
from algorithm_3_momentum_detector import MomentumDetector
from algorithm_4_trend_detector import TrendDetector
from algorithm_5_risk_management import DynamicScoreSystem

# Definir a estrutura Candle (copiada de main.py para evitar dependência circular)
class Candle:
    def __init__(self, open_time_ms: int, open: float, high: float, low: float, close: float, volume: float, close_time_ms: int):
        self.open_time_ms = open_time_ms
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.close_time_ms = close_time_ms

def _convert_candles_to_dict(candles: List[Candle]) -> List[Dict[str, Any]]:
    """Converte a lista de objetos Candle para a lista de dicionários esperada pelos algoritmos."""
    converted = []
    for c in candles:
        converted.append({
            'datetime': datetime.fromtimestamp(c.close_time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'open_time_ms': c.open_time_ms,
            'close_time_ms': c.close_time_ms,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume,
        })
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
        if result["score_analysis"]:
            result["score_analysis"]["quality"] = result["score_analysis"]["quality"].value
            result["score_analysis"]["risk_params"] = asdict(result["score_analysis"]["risk_params"])
            result["score_analysis"]["management"] = asdict(result["score_analysis"]["management"])
        
        # Limpar enum values para strings no trend_analysis
        if result["trend_analysis"]:
            result["trend_analysis"]["trend_state"] = result["trend_analysis"]["trend_state"].value
            result["trend_analysis"]["pullback_zone"] = result["trend_analysis"]["pullback_zone"].value
        
        # Limpar enum values para strings nos setups
        if result["long_setup"]:
            result["long_setup"]["strength"] = result["long_setup"]["strength"].value
        if result["short_setup"]:
            result["short_setup"]["strength"] = result["short_setup"]["strength"].value
            
        # Limpar enum values para strings no momentum_analysis
        for sig in result["momentum_analysis"]:
            sig["signal_type"] = sig["signal_type"].value

        return result

# Helper para converter dataclass para dict (simples, sem recursão profunda)
def asdict(obj):
    if hasattr(obj, '__dict__'):
        data = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                data[key] = asdict(value)
        return data
    elif isinstance(obj, list):
        return [asdict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: asdict(value) for key, value in obj.items()}
    elif hasattr(obj, 'value'):
        return obj.value
    return obj

# Sobrescrever o asdict do módulo para usar o nosso helper
from algorithm_1_confluence_long import SetupSignal
from algorithm_2_confluence_short import SetupSignal as ShortSetupSignal
from algorithm_3_momentum_detector import MomentumSignal
from algorithm_4_trend_detector import TrendAnalysis
from algorithm_5_risk_management import SetupAnalysis as ScoreSetupAnalysis

# Substituir o asdict interno dos módulos para usar o nosso
# Isso é necessário porque os módulos foram escritos sem o dataclasses.asdict
# e usam um helper simples. Vou garantir que o nosso helper funcione.
# Como os módulos usam dataclasses, vou usar o dataclasses.asdict real se estiver disponível,
# ou o meu helper se não estiver. Como o dataclasses não está importado nos módulos,
# vou confiar no meu helper simples.

# Para evitar erros de importação, vou garantir que as classes dos algoritmos
# não dependam de dataclasses.asdict, mas sim de um asdict local ou que o resultado
# seja um objeto simples que eu possa processar.

# O jeito mais seguro é importar o dataclasses.asdict no topo e usá-lo.
from dataclasses import asdict as dataclass_asdict

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
    """Tenta usar dataclasses.asdict e limpa Enums."""
    try:
        data = dataclass_asdict(obj)
        return _clean_dataclass_dict(data)
    except TypeError:
        # Se não for um dataclass, retorna o objeto
        return obj

# Re-executar a análise com o asdict seguro
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

        candles_dict = _convert_candles_to_dict(candles)
        
        long_setup = self.long_detector.detect_setup(candles_dict)
        short_setup = self.short_detector.detect_setup(candles_dict)
        momentum_signals = self.momentum_detector.analyze(candles_dict)
        trend_analysis = self.trend_detector.analyze(candles_dict)
        score_analysis = self.score_system.analyze(candles_dict, capital=capital)

        result = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "candles_analyzed": len(candles),
            "long_setup": _safe_asdict(long_setup),
            "short_setup": _safe_asdict(short_setup),
            "momentum_analysis": [_safe_asdict(s) for s in momentum_signals],
            "trend_analysis": _safe_asdict(trend_analysis),
            "score_analysis": _safe_asdict(score_analysis),
        }
        
        return result

# Instanciar o analisador para uso no main.py
analyzer = ScalpingAnalyzer()
