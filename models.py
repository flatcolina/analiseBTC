from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

@dataclass
class Candle:
    """Representa um candle de preço com indicadores técnicos."""
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float
    ignore: float
    # Indicadores Técnicos
    vwap: Optional[float] = None
    ema9: Optional[float] = None
    ema21: Optional[float] = None
    ema55: Optional[float] = None
    ema200: Optional[float] = None
    atr: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None

@dataclass
class SetupAnalysis:
    """Resultado dos Algoritmos 1 e 2 (Confluência Completa)."""
    is_ready: bool
    direction: Literal["LONG", "SHORT"]
    strength: Literal["FORTE", "MODERADO", "FRACO"]
    entry_price: float
    conditions: Dict[str, bool]
    comment: str

@dataclass
class MomentumSignal:
    """Resultado do Algoritmo 3 (Detector de Momentum)."""
    signal_type: str # "DIVERGENCIA_ALTA", "CRUZAMENTO_MACD", "RSI_SOBRECOMPRA"
    direction: Literal["LONG", "SHORT"]
    strength: int # 0-100
    rsi_value: float
    macd_value: float
    macd_signal: float
    macd_histogram: float
    description: str

@dataclass
class TrendAnalysis:
    """Resultado do Algoritmo 4 (Detector de Tendência)."""
    trend_state: Literal["TENDENCIA_ALTA", "TENDENCIA_BAIXA", "CONSOLIDACAO"]
    trend_strength: int # 0-100
    ema_alignment: Literal["ALINHADO_ALTA", "ALINHADO_BAIXA", "CRUZADO"]
    vwap_bias: Literal["ACIMA", "ABAIXO", "NEUTRO"]
    pullback_zone: Literal["EMA9", "EMA21", "VWAP", "NENHUMA"]
    entry_quality: int # 0-100
    recommendation: str
    price: float
    vwap: float
    ema9: float
    ema21: float
    ema55: float

@dataclass
class RiskParams:
    """Parâmetros de Risco (Algoritmo 5)."""
    risk_amount: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size_suggestion: float # % do capital

@dataclass
class ManagementParams:
    """Parâmetros de Gerenciamento de Risco (Algoritmo 5)."""
    move_to_breakeven_at: float
    partial_exit_at: float

@dataclass
class ScoreAnalysis:
    """Resultado do Algoritmo 5 (Score Dinâmico)."""
    score: int # 0-100
    quality: Literal["EXCELENTE", "BOM", "MEDIO", "RUIM"]
    direction: Literal["LONG", "SHORT", "NEUTRO"]
    recommendation: str
    update_frequency_seconds: int
    score_breakdown: Dict[str, int]
    risk_params: RiskParams
    management: ManagementParams

@dataclass
class FullAnalysis:
    """Estrutura completa para o endpoint /api/full_analysis."""
    snapshot: Candle
    long_setup: SetupAnalysis
    short_setup: SetupAnalysis
    momentum_analysis: List[MomentumSignal]
    trend_analysis: TrendAnalysis
    score_analysis: ScoreAnalysis
