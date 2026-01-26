# -*- coding: utf-8 -*-
"""scalping_analyzer.py

Orquestrador dos 5 algoritmos.

Objetivos:
- Sempre devolver estruturas estáveis para o frontend (mesmo quando não há sinal).
- Expor configuração em tempo real via API (/api/config), **apenas em memória**.

Obs.: o main.py já calcula e injeta indicadores (EMA/RSI/MACD/ATR/VWAP) na Candle.
"""

from __future__ import annotations

from dataclasses import asdict as dataclass_asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from models import Candle, FullAnalysis

from algorithm_3_momentum_detector import MomentumDetector
from algorithm_4_trend_detector import TrendDetector
from algorithm_5_risk_management import DynamicScoreSystem


# ---------------------------------------------------------------------------
# Configuração (em memória)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "confluence": {
        # Tolerância usada para considerar "pullback" (distância percentual da EMA)
        "pullback_tolerance": 0.002,  # 0.2%
        "rsi_long_min": 50.0,
        "rsi_short_max": 50.0,
        # Apenas para referência / score (o Alg 5 é quem calcula SL/TP)
        "atr_multiplier_sl": 1.5,
        "risk_reward_ratio": 1.5,
    },
    "momentum": {
        "rsi_period": 14,
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "divergence_lookback": 14,
    },
    "trend": {
        "ema_periods": [9, 21, 55],
        "pullback_tolerance": 0.003,
        "strong_trend_spread": 0.005,
    },
    "score": {
        "ema_periods": [9, 21, 55],
        "rsi_period": 14,
        "macd_params": [12, 26, 9],
        "atr_period": 14,
        "atr_multiplier_sl": 1.5,
        "risk_reward_1": 1.0,
        "risk_reward_2": 1.5,
        "risk_reward_3": 2.0,
        "breakeven_trigger_pct": 0.5,
        "partial_exit_pct": 0.5,
        "max_risk_per_trade": 0.02,
    },
}


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _clamp_float(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return max(lo, min(hi, x))


def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        x = int(v)
    except Exception:
        return default
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Serialização segura
# ---------------------------------------------------------------------------


def _clean_enum_like(v: Any) -> Any:
    if hasattr(v, "value") and not isinstance(v, (str, int, float, bool, dict, list, tuple)):
        try:
            return v.value
        except Exception:
            return str(v)
    return v


def _safe_asdict(obj: Any) -> Any:
    if obj is None:
        return None

    if is_dataclass(obj):
        data = dataclass_asdict(obj)
    elif isinstance(obj, dict):
        data = obj
    elif isinstance(obj, list):
        return [_safe_asdict(x) for x in obj]
    else:
        return _clean_enum_like(obj)

    def _walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        return _clean_enum_like(x)

    return _walk(data)


# ---------------------------------------------------------------------------
# Traduções (para o frontend ficar 100% em PT)
# ---------------------------------------------------------------------------


def _dir_pt(direction: str) -> str:
    d = (direction or "").upper()
    if d == "LONG":
        return "COMPRA"
    if d == "SHORT":
        return "VENDA"
    return direction


MOMENTUM_TYPE_PT = {
    "rsi_oversold_exit": "RSI saindo de sobrevenda",
    "rsi_overbought_exit": "RSI saindo de sobrecompra",
    "rsi_centerline_cross_up": "RSI cruzou 50 para cima",
    "rsi_centerline_cross_down": "RSI cruzou 50 para baixo",
    "macd_bullish_cross": "MACD cruzou acima do sinal",
    "macd_bearish_cross": "MACD cruzou abaixo do sinal",
    "bullish_divergence": "Divergência de alta",
    "bearish_divergence": "Divergência de baixa",
    "histogram_reversal_up": "Histograma virou para cima",
    "histogram_reversal_down": "Histograma virou para baixo",
}


TREND_STATE_PT = {
    "strong_uptrend": "Alta forte",
    "uptrend": "Alta",
    "ranging": "Lateralizado",
    "downtrend": "Baixa",
    "strong_downtrend": "Baixa forte",
    "neutral": "Neutro",
}


BIAS_PT = {
    "bullish": "Acima da VWAP",
    "bearish": "Abaixo da VWAP",
    "neutral": "Neutro",
}


ALIGN_PT = {
    "bullish": "Alinhado (alta)",
    "bearish": "Alinhado (baixa)",
    "mixed": "Misto",
    "neutral": "Neutro",
}


PULLBACK_PT = {
    "ema_fast": "Próximo da EMA rápida",
    "ema_mid": "Próximo da EMA média",
    "vwap": "Próximo da VWAP",
    "none": "Nenhuma",
}


QUALITY_PT = {
    "excellent": "Excelente",
    "good": "Bom",
    "acceptable": "Aceitável",
    "poor": "Fraco",
    "avoid": "Evitar",
}


# ---------------------------------------------------------------------------
# Avaliadores (sempre retornam objeto, com condições)
# ---------------------------------------------------------------------------


def _pct_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if b == 0:
        return None
    return abs(a - b) / abs(b)


def _confluence_setup(last: Candle, direction: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Cria um setup estável (mesmo sem sinal) usando indicadores já calculados."""

    # Indicadores básicos
    close = last.close
    vwap = getattr(last, "vwap", None)
    ema9 = getattr(last, "ema9", None)
    ema21 = getattr(last, "ema21", None)
    ema55 = getattr(last, "ema55", None)
    rsi = getattr(last, "rsi", None)
    macd = getattr(last, "macd", None)
    macd_sig = getattr(last, "macd_signal", None)

    tol = _clamp_float(cfg.get("pullback_tolerance"), 0.0001, 0.02, DEFAULT_CONFIG["confluence"]["pullback_tolerance"])
    rsi_long_min = _clamp_float(cfg.get("rsi_long_min"), 1.0, 99.0, DEFAULT_CONFIG["confluence"]["rsi_long_min"])
    rsi_short_max = _clamp_float(cfg.get("rsi_short_max"), 1.0, 99.0, DEFAULT_CONFIG["confluence"]["rsi_short_max"])

    is_long = (direction or "").lower() in ("long", "compra")

    # Condições
    conds: Dict[str, bool] = {}

    # VWAP
    if vwap is None:
        conds["VWAP disponível"] = False
        above_vwap = False
        below_vwap = False
    else:
        above_vwap = close > vwap
        below_vwap = close < vwap
        conds["Preço acima da VWAP" if is_long else "Preço abaixo da VWAP"] = above_vwap if is_long else below_vwap

    # EMAs
    if None in (ema9, ema21, ema55):
        conds["EMAs disponíveis"] = False
        ema_ok = False
    else:
        ema_ok = (ema9 > ema21 > ema55) if is_long else (ema9 < ema21 < ema55)
        conds["EMAs alinhadas (alta)" if is_long else "EMAs alinhadas (baixa)"] = ema_ok

    # RSI
    if rsi is None:
        conds["RSI disponível"] = False
        rsi_ok = False
    else:
        rsi_ok = (rsi >= rsi_long_min) if is_long else (rsi <= rsi_short_max)
        conds["RSI favorável"] = rsi_ok

    # MACD
    if macd is None or macd_sig is None:
        conds["MACD disponível"] = False
        macd_ok = False
    else:
        macd_ok = (macd > macd_sig) if is_long else (macd < macd_sig)
        conds["MACD confirma"] = macd_ok

    # Pullback
    if ema9 is None or ema21 is None:
        conds["Zona de pullback"] = False
        pullback_ok = False
    else:
        # entre EMA9 e EMA21 ou bem perto da EMA9
        between = (ema9 <= close <= ema21) or (ema21 <= close <= ema9)
        near_ema9 = (_pct_diff(close, ema9) or 999) <= tol
        pullback_ok = between or near_ema9
        conds["Zona de pullback"] = pullback_ok

    met = sum(1 for v in conds.values() if v)
    total = len(conds)

    if met == total:
        strength = "FORTE"
    elif met >= max(1, total - 1):
        strength = "MÉDIO"
    else:
        strength = "FRACO"

    is_ready = met == total and total > 0

    # Comentário
    if is_ready:
        comment = "Setup confirmado: todas as condições estão alinhadas."
    else:
        missing = [k for k, v in conds.items() if not v]
        comment = "Aguardando confirmação. Faltando: " + ", ".join(missing[:6]) + ("..." if len(missing) > 6 else "")

    return {
        "is_ready": is_ready,
        "direction": "COMPRA" if is_long else "VENDA",
        "strength": strength,
        "entry_price": close,
        "conditions": conds,
        "comment": comment,
    }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ScalpingAnalyzer:
    """Orquestra os 5 algoritmos e retorna a análise completa."""

    def __init__(self) -> None:
        self._config: Dict[str, Any] = _deep_merge({}, DEFAULT_CONFIG)
        self._rebuild_algorithms()

    def _rebuild_algorithms(self) -> None:
        m = self._config.get("momentum", {})
        t = self._config.get("trend", {})
        s = self._config.get("score", {})

        self.momentum_detector = MomentumDetector(
            rsi_period=_clamp_int(m.get("rsi_period"), 2, 200, DEFAULT_CONFIG["momentum"]["rsi_period"]),
            rsi_oversold=_clamp_float(m.get("rsi_oversold"), 1, 49, DEFAULT_CONFIG["momentum"]["rsi_oversold"]),
            rsi_overbought=_clamp_float(m.get("rsi_overbought"), 51, 99, DEFAULT_CONFIG["momentum"]["rsi_overbought"]),
            macd_fast=_clamp_int(m.get("macd_fast"), 2, 100, DEFAULT_CONFIG["momentum"]["macd_fast"]),
            macd_slow=_clamp_int(m.get("macd_slow"), 3, 200, DEFAULT_CONFIG["momentum"]["macd_slow"]),
            macd_signal=_clamp_int(m.get("macd_signal"), 2, 60, DEFAULT_CONFIG["momentum"]["macd_signal"]),
            divergence_lookback=_clamp_int(m.get("divergence_lookback"), 5, 200, DEFAULT_CONFIG["momentum"]["divergence_lookback"]),
        )

        ema_periods = t.get("ema_periods") or DEFAULT_CONFIG["trend"]["ema_periods"]
        if isinstance(ema_periods, (list, tuple)) and len(ema_periods) == 3:
            ema_periods = (int(ema_periods[0]), int(ema_periods[1]), int(ema_periods[2]))
        else:
            ema_periods = (9, 21, 55)

        self.trend_detector = TrendDetector(
            ema_periods=ema_periods,
            pullback_tolerance=_clamp_float(t.get("pullback_tolerance"), 0.0001, 0.05, DEFAULT_CONFIG["trend"]["pullback_tolerance"]),
            strong_trend_spread=_clamp_float(t.get("strong_trend_spread"), 0.0001, 0.1, DEFAULT_CONFIG["trend"]["strong_trend_spread"]),
        )

        s_ema = s.get("ema_periods") or DEFAULT_CONFIG["score"]["ema_periods"]
        if isinstance(s_ema, (list, tuple)) and len(s_ema) == 3:
            s_ema = (int(s_ema[0]), int(s_ema[1]), int(s_ema[2]))
        else:
            s_ema = (9, 21, 55)

        macd_params = s.get("macd_params") or DEFAULT_CONFIG["score"]["macd_params"]
        if isinstance(macd_params, (list, tuple)) and len(macd_params) == 3:
            macd_params = (int(macd_params[0]), int(macd_params[1]), int(macd_params[2]))
        else:
            macd_params = (12, 26, 9)

        self.score_system = DynamicScoreSystem(
            ema_periods=s_ema,
            rsi_period=_clamp_int(s.get("rsi_period"), 2, 200, DEFAULT_CONFIG["score"]["rsi_period"]),
            macd_params=macd_params,
            atr_period=_clamp_int(s.get("atr_period"), 2, 200, DEFAULT_CONFIG["score"]["atr_period"]),
            atr_multiplier_sl=_clamp_float(s.get("atr_multiplier_sl"), 0.1, 10.0, DEFAULT_CONFIG["score"]["atr_multiplier_sl"]),
            risk_reward_1=_clamp_float(s.get("risk_reward_1"), 0.2, 10.0, DEFAULT_CONFIG["score"]["risk_reward_1"]),
            risk_reward_2=_clamp_float(s.get("risk_reward_2"), 0.2, 10.0, DEFAULT_CONFIG["score"]["risk_reward_2"]),
            risk_reward_3=_clamp_float(s.get("risk_reward_3"), 0.2, 10.0, DEFAULT_CONFIG["score"]["risk_reward_3"]),
            breakeven_trigger_pct=_clamp_float(s.get("breakeven_trigger_pct"), 0.1, 1.0, DEFAULT_CONFIG["score"]["breakeven_trigger_pct"]),
            partial_exit_pct=_clamp_float(s.get("partial_exit_pct"), 0.05, 1.0, DEFAULT_CONFIG["score"]["partial_exit_pct"]),
            max_risk_per_trade=_clamp_float(s.get("max_risk_per_trade"), 0.001, 0.2, DEFAULT_CONFIG["score"]["max_risk_per_trade"]),
        )

    # ---------------------------
    # Config API
    # ---------------------------

    def get_config(self) -> Dict[str, Any]:
        return _deep_merge({}, self._config)

    def set_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        # patch pode vir como {config: {...}} ou direto {...}
        if isinstance(patch, dict) and "config" in patch and isinstance(patch["config"], dict):
            patch = patch["config"]
        self._config = _deep_merge(self._config, patch or {})
        self._rebuild_algorithms()
        return self.get_config()

    # ---------------------------
    # Análise
    # ---------------------------

    def get_full_analysis(self, candles: List[Candle], capital: float = 10000.0) -> Dict[str, Any]:
        if len(candles) < 60:
            return {"error": "Insufficient data (need at least 60 candles)"}

        candles_dict = [
            {
                **dataclass_asdict(c),
                "datetime": datetime.fromtimestamp(c.close_time / 1000).strftime("%Y-%m-%d %H:%M:%S"),
            }
            for c in candles
        ]

        last = candles[-1]

        # Confluência (sempre objeto)
        long_setup = _confluence_setup(last, "long", self._config.get("confluence", {}))
        short_setup = _confluence_setup(last, "short", self._config.get("confluence", {}))

        # Momentum
        momentum_signals = self.momentum_detector.analyze(candles_dict) or []
        mom_out: List[Dict[str, Any]] = []
        for s in momentum_signals:
            sd = _safe_asdict(s)
            # normaliza
            if isinstance(sd, dict):
                st = str(sd.get("signal_type"))
                sd["signal_type"] = MOMENTUM_TYPE_PT.get(st, st)
                sd["direction"] = _dir_pt(str(sd.get("direction", "")))
            mom_out.append(sd)

        # Tendência
        trend_analysis = _safe_asdict(self.trend_detector.analyze(candles_dict))
        if isinstance(trend_analysis, dict):
            trend_analysis["trend_state"] = TREND_STATE_PT.get(str(trend_analysis.get("trend_state")), str(trend_analysis.get("trend_state")))
            trend_analysis["ema_alignment"] = ALIGN_PT.get(str(trend_analysis.get("ema_alignment")), str(trend_analysis.get("ema_alignment")))
            trend_analysis["vwap_bias"] = BIAS_PT.get(str(trend_analysis.get("vwap_bias")), str(trend_analysis.get("vwap_bias")))
            trend_analysis["pullback_zone"] = PULLBACK_PT.get(str(trend_analysis.get("pullback_zone")), str(trend_analysis.get("pullback_zone")))

        # Score/Risco
        score_analysis = _safe_asdict(self.score_system.analyze(candles_dict, capital=capital))
        if isinstance(score_analysis, dict):
            score_analysis["direction"] = _dir_pt(str(score_analysis.get("direction", "")))
            q = str(score_analysis.get("quality"))
            score_analysis["quality"] = QUALITY_PT.get(q, q)

        full = FullAnalysis(
            snapshot=last,
            long_setup=long_setup,
            short_setup=short_setup,
            momentum_analysis=mom_out,
            trend_analysis=trend_analysis,
            score_analysis=score_analysis,
        )

        return dataclass_asdict(full)

    def analyze_all(self, candles: List[Candle], capital: float = 10000.0) -> Dict[str, Any]:
        analysis = self.get_full_analysis(candles, capital)
        if "error" in analysis:
            return analysis
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "candles_analyzed": len(candles),
            **analysis,
        }


# Instância única
analyzer = ScalpingAnalyzer()


def analyze_all(candles: List[Candle], capital: float = 10000.0) -> Dict[str, Any]:
    return analyzer.analyze_all(candles, capital)
