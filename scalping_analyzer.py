# -*- coding: utf-8 -*-
"""scalping_analyzer.py

Módulo de Análise de Scalping Integrada.

Integra os 5 algoritmos do projeto para gerar uma saída unificada.
Este módulo expõe:

- `analyzer`: instância única de `ScalpingAnalyzer`
- `analyze_all(...)`: compatibilidade com chamadas antigas

Observação: o arquivo anterior estava com blocos duplicados e indentação quebrada
(funções e até uma segunda classe dentro de um método), causando `IndentationError`
no deploy.
"""

from __future__ import annotations

from dataclasses import asdict as dataclass_asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List

from models import Candle, FullAnalysis

from algorithm_1_confluence_long import ConfluenceLongDetector
from algorithm_2_confluence_short import ConfluenceShortDetector
from algorithm_3_momentum_detector import MomentumDetector
from algorithm_4_trend_detector import TrendDetector
from algorithm_5_risk_management import DynamicScoreSystem


def _convert_candles_to_dict(candles: List[Candle]) -> List[Dict[str, Any]]:
    """Converte `Candle` (dataclass) para o formato dict esperado pelos algoritmos."""
    out: List[Dict[str, Any]] = []
    for c in candles:
        d = dataclass_asdict(c)
        # Alguns algoritmos esperam a chave `datetime` string.
        d["datetime"] = datetime.fromtimestamp(c.close_time / 1000).strftime("%Y-%m-%d %H:%M:%S")
        out.append(d)
    return out


def _clean_enum_like(v: Any) -> Any:
    """Converte objetos estilo Enum (possuem `.value`) para valores serializáveis."""
    if hasattr(v, "value") and not isinstance(v, (str, int, float, bool, dict, list, tuple)):
        try:
            return v.value
        except Exception:
            return str(v)
    return v


def _safe_asdict(obj: Any) -> Any:
    """Converte dataclasses -> dict e limpa enums/estruturas aninhadas.

    Alguns algoritmos podem retornar objetos não-dataclass; nesse caso retornamos o próprio objeto.
    """
    if obj is None:
        return None

    # Dataclass -> dict
    if is_dataclass(obj):
        data = dataclass_asdict(obj)
    elif isinstance(obj, dict):
        data = obj
    elif isinstance(obj, list):
        return [_safe_asdict(x) for x in obj]
    else:
        # Tenta serializar objetos simples / enums
        return _clean_enum_like(obj)

    # Limpeza recursiva
    def _walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        return _clean_enum_like(x)

    return _walk(data)


class ScalpingAnalyzer:
    """Orquestra os 5 algoritmos e retorna a análise completa."""

    def __init__(self) -> None:
        self.long_detector = ConfluenceLongDetector()
        self.short_detector = ConfluenceShortDetector()
        self.momentum_detector = MomentumDetector()
        self.trend_detector = TrendDetector()
        self.score_system = DynamicScoreSystem()

    def get_full_analysis(self, candles: List[Candle], capital: float = 10000.0) -> Dict[str, Any]:
        """Retorna a estrutura compatível com o response_model `FullAnalysis` do main.py."""
        if len(candles) < 60:
            return {"error": "Insufficient data (need at least 60 candles)"}

        candles_dict = _convert_candles_to_dict(candles)

        long_setup = self.long_detector.detect_setup(candles_dict)
        short_setup = self.short_detector.detect_setup(candles_dict)
        momentum_signals = self.momentum_detector.analyze(candles_dict)
        trend_analysis = self.trend_detector.analyze(candles_dict)
        score_analysis = self.score_system.analyze(candles_dict, capital=capital)

        full = FullAnalysis(
            snapshot=candles[-1],  # última vela já com indicadores computados no main.py
            long_setup=_safe_asdict(long_setup),
            short_setup=_safe_asdict(short_setup),
            momentum_analysis=[_safe_asdict(s) for s in (momentum_signals or [])],
            trend_analysis=_safe_asdict(trend_analysis),
            score_analysis=_safe_asdict(score_analysis),
        )

        return dataclass_asdict(full)

    def analyze_all(self, candles: List[Candle], capital: float = 10000.0) -> Dict[str, Any]:
        """Compatibilidade: devolve um dict simples com timestamp + blocos de análise."""
        analysis = self.get_full_analysis(candles, capital)
        if "error" in analysis:
            return analysis
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "candles_analyzed": len(candles),
            **analysis,
        }


# Instância única para import no main.py
analyzer = ScalpingAnalyzer()


def analyze_all(candles: List[Candle], capital: float = 10000.0) -> Dict[str, Any]:
    """Atalho de módulo (compatibilidade)."""
    return analyzer.analyze_all(candles, capital)
