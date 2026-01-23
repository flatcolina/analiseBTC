from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

Direction = Literal["LONG", "SHORT"]


@dataclass
class AnalysisSnapshot:
    """Snapshot leve usado pelos cenários/monitoramento do backend.

    Este snapshot é menor que `Candle` e contém somente os campos que o motor
    de cenários (pullback/breakout) e os endpoints antigos precisam.
    """

    ts_ms: int
    symbol: str
    interval: str
    price: float

    vwap: Optional[float] = None
    ema9: Optional[float] = None
    ema21: Optional[float] = None
    ema55: Optional[float] = None
    ema200: Optional[float] = None
    rsi14: Optional[float] = None
    atr14: Optional[float] = None

    avg_vol20: Optional[float] = None
    recent_high: Optional[float] = None
    recent_low: Optional[float] = None

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
    # Indicadores Técnicos (Adicionados para os 5 Algoritmos)
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
class TradeState:
    """Estado de uma operação ativa, serializável via `asdict()`.

    Campos cobrem o que `main.py` e `trade_intelligence.py` utilizam.
    """

    cycle_id: int
    scenario_key: str
    scenario_kind: str
    direction: Direction

    entry_time_ms: int
    entry_raw: float
    entry_exec: float
    qty_btc: float

    tp_price: float
    sl_price: float
    atr_at_entry: float

    last_price: float

    # Gestão
    last_manage_ms: int = 0
    be_moved: bool = False
    tp_extended: bool = False

    # Diagnósticos
    best_fav_price: float = 0.0
    worst_adv_price: float = 0.0
    mfe_gross_usd: float = 0.0
    mae_gross_usd: float = 0.0

    # Distâncias/closest
    closest_tp_price: float = 0.0
    closest_tp_dist: float = 0.0
    closest_tp_ts_ms: int = 0

    closest_sl_price: float = 0.0
    closest_sl_dist: float = 0.0
    closest_sl_ts_ms: int = 0

    # Indicadores coletados durante o trade
    entry_indicators: Dict[str, Any] = field(default_factory=dict)
    indicator_samples: List[Dict[str, Any]] = field(default_factory=list)


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
