# -*- coding: utf-8 -*-
"""b3_backtest.py

Backtest simples (offline) para B3 / mini-índice a partir de um CSV.

Objetivo (caminho C):
- Usuário fornece um arquivo (ticks ou candles) e o sistema:
  1) agrega/resample para 30s (OHLCV)
  2) calcula indicadores (EMA/RSI/MACD/ATR/VWAP)
  3) roda as análises (mesmos 5 algoritmos)
  4) abre trades simulados e registra um diário a cada 30s
  5) permite múltiplos trades em paralelo (sem sobrescrever o anterior)

Formato de CSV aceito (auto-detect):
1) Tick: precisa ter `timestamp` (ou `ts`) e `price` (ou `preco`).
   Volume é opcional (`volume`, `vol`).
2) Candle: precisa ter `timestamp` e `open,high,low,close`.
   Volume é opcional.

`timestamp` pode ser:
- epoch em ms ou s
- ISO 8601 (ex: 2026-01-23 10:01:00)

Obs.: Este módulo não sabe baixar dados da B3. Ele processa um arquivo que
você já exportou do seu fornecedor (Profit/Neologica, Tryd, MetaTrader, etc).
"""

from __future__ import annotations

import csv
import io
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from models import Candle
from indicators import ema_series, rsi_series, atr_wilder_series, avg_volume, vwap_rolling, macd_series


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace(",", ".")
        if not s:
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _to_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return int(default)
        if isinstance(v, bool):
            return int(default)
        if isinstance(v, int):
            return int(v)
        s = str(v).strip()
        if not s:
            return int(default)
        return int(float(s))
    except Exception:
        return int(default)


def _parse_timestamp_ms(v: Any) -> Optional[int]:
    """Aceita epoch s/ms ou ISO e devolve epoch em ms."""
    if v is None:
        return None
    # numérico
    if isinstance(v, (int, float)):
        x = float(v)
        if x <= 0:
            return None
        # heurística: > 10^12 é ms
        if x > 1e12:
            return int(x)
        # senão, segundos
        if x > 1e9:
            return int(x * 1000)
        return int(x * 1000)

    s = str(v).strip()
    if not s:
        return None

    # se é número em string
    if s.replace(".", "", 1).isdigit():
        return _parse_timestamp_ms(float(s))

    # ISO / formatos comuns
    # tenta vários
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            # assume horário local do usuário? Para backtest, convertemos para UTC.
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def read_csv_rows(raw_bytes: bytes, max_rows: int = 2_000_000) -> List[Dict[str, Any]]:
    """Lê CSV em memória e devolve lista de dicts (até max_rows)."""
    text = raw_bytes.decode("utf-8", errors="ignore")
    # tenta detectar delimitador
    sample = text[:4096]
    dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        out.append({k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in (row or {}).items() if k})
    return out


def _detect_schema(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "unknown"
    keys = set(rows[0].keys())
    # candle
    if {"open", "high", "low", "close"}.issubset(keys):
        return "candle"
    # tick
    if any(k in keys for k in ("price", "preco", "last", "close")):
        return "tick"
    return "unknown"


def to_30s_bars(rows: List[Dict[str, Any]]) -> List[Candle]:
    """Converte tick/candle em candles de 30s (OHLCV)."""
    schema = _detect_schema(rows)
    if schema == "unknown":
        return []

    # normaliza para uma sequência de "eventos" com ts_ms e ohlc
    events: List[Tuple[int, float, float, float, float, float]] = []
    # (ts_ms, open, high, low, close, volume)

    for r in rows:
        ts = _parse_timestamp_ms(r.get("timestamp") or r.get("ts") or r.get("time") or r.get("datetime") or r.get("date"))
        if ts is None:
            continue
        if schema == "candle":
            o = _to_float(r.get("open"))
            h = _to_float(r.get("high"))
            l = _to_float(r.get("low"))
            c = _to_float(r.get("close"))
            v = _to_float(r.get("volume") or r.get("vol") or 0.0)
            events.append((ts, o, h, l, c, v))
        else:
            p = _to_float(r.get("price") or r.get("preco") or r.get("last") or r.get("close"))
            v = _to_float(r.get("volume") or r.get("vol") or 0.0)
            events.append((ts, p, p, p, p, v))

    if not events:
        return []

    # ordena por tempo
    events.sort(key=lambda x: x[0])

    bucket_ms = 30_000
    bars: List[Candle] = []

    cur_bucket: Optional[int] = None
    o = h = l = c = v = 0.0
    first_ts = last_ts = 0

    def _flush() -> None:
        nonlocal o, h, l, c, v, first_ts, last_ts, cur_bucket
        if cur_bucket is None:
            return
        open_time = cur_bucket
        close_time = cur_bucket + bucket_ms - 1
        bars.append(
            Candle(
                open_time=open_time,
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v),
                close_time=close_time,
                quote_asset_volume=0.0,
                number_of_trades=0,
                taker_buy_base_asset_volume=0.0,
                taker_buy_quote_asset_volume=0.0,
                ignore=0.0,
            )
        )
        cur_bucket = None

    for ts, eo, eh, el, ec, ev in events:
        b = (ts // bucket_ms) * bucket_ms
        if cur_bucket is None:
            cur_bucket = b
            o, h, l, c = eo, eh, el, ec
            v = ev
            first_ts = ts
            last_ts = ts
            continue
        if b != cur_bucket:
            _flush()
            cur_bucket = b
            o, h, l, c = eo, eh, el, ec
            v = ev
            first_ts = ts
            last_ts = ts
            continue
        # mesmo bucket
        h = max(h, eh)
        l = min(l, el)
        c = ec
        v += ev
        last_ts = ts

    _flush()

    return bars


def compute_indicators(candles: List[Candle]) -> List[Candle]:
    closes = [c.close for c in candles]
    ema9s = ema_series(closes, 9)
    ema21s = ema_series(closes, 21)
    ema55s = ema_series(closes, 55)
    ema200s = ema_series(closes, 200)
    rsis = rsi_series(candles, 14)
    atrs = atr_wilder_series(candles, 14)
    vwaps = [vwap_rolling(candles[: i + 1], 100) for i in range(len(candles))]
    macd_line, signal_line, histogram = macd_series(closes)

    for i, c in enumerate(candles):
        c.ema9 = ema9s[i]
        c.ema21 = ema21s[i]
        c.ema55 = ema55s[i]
        c.ema200 = ema200s[i]
        c.rsi = rsis[i]
        c.atr = atrs[i]
        c.vwap = vwaps[i]
        c.macd = macd_line[i]
        c.macd_signal = signal_line[i]
        c.macd_hist = histogram[i]
    return candles


def ms_to_iso(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ms / 1000.0)) + f".{ms%1000:03d}Z"


def calc_gross_pnl_b3(entry: float, exit_: float, direction: str, contracts: int, point_value_brl: float) -> float:
    """PnL bruto (sem custos) em BRL.

    Exemplo WIN: 1 ponto = R$ 0,20 (point_value_brl=0.2)
    """
    if contracts <= 0:
        contracts = 1
    dir_up = (direction or "").upper() in ("LONG", "COMPRA", "BUY")
    points = (exit_ - entry) if dir_up else (entry - exit_)
    return float(points) * float(point_value_brl) * float(contracts)


def calc_fees_b3(
    *,
    contracts: int,
    brokerage_brl_per_contract_per_side: float,
    exchange_fee_brl_per_contract_per_side: float,
    iss_pct_on_brokerage: float = 0.0,
    sides: int = 2,
) -> float:
    """Custos aproximados (BRL).

    - brokerage_brl_per_contract_per_side: corretagem por contrato por lado (entrada ou saída)
    - exchange_fee_brl_per_contract_per_side: emolumentos/taxas por contrato por lado
    - iss_pct_on_brokerage: ISS (0.05 = 5%) aplicado *somente* sobre a corretagem
    """
    if contracts <= 0:
        contracts = 1
    sides = max(1, int(sides))
    b = float(brokerage_brl_per_contract_per_side) * float(contracts) * float(sides)
    ex = float(exchange_fee_brl_per_contract_per_side) * float(contracts) * float(sides)
    iss = float(brokerage_brl_per_contract_per_side) * float(contracts) * float(sides) * float(iss_pct_on_brokerage)
    return float(b + ex + iss)


def _round_to_tick(price: float, tick_size_points: float, mode: str = "nearest") -> float:
    """Arredonda preço para o múltiplo do tick (em pontos)."""
    t = float(tick_size_points or 0.0)
    if t <= 0:
        return float(price)
    x = float(price) / t
    if mode == "up":
        return math.ceil(x) * t
    if mode == "down":
        return math.floor(x) * t
    # nearest
    return round(x) * t


def _apply_execution_friction(
    *,
    price: float,
    direction: str,
    side: str,
    tick_size_points: float,
    slippage_ticks: int,
) -> float:
    """Aplica arredondamento de tick + slippage (sempre adverso)."""
    dir_ = (direction or "").upper()
    side_ = (side or "").upper()  # ENTRY/EXIT
    t = float(tick_size_points or 0.0)

    # 1) tick rounding (adverso)
    if t > 0:
        if side_ == "ENTRY":
            # LONG compra: pior é pagar MAIS (ceil). SHORT venda: pior é vender MENOS (floor)
            mode = "up" if dir_ == "LONG" else "down"
            price = _round_to_tick(price, t, mode=mode)
        else:
            # EXIT: LONG vendendo: pior é vender MENOS (floor). SHORT comprando: pior é comprar MAIS (ceil)
            mode = "down" if dir_ == "LONG" else "up"
            price = _round_to_tick(price, t, mode=mode)

    # 2) slippage (em ticks)
    slip_points = float(max(0, int(slippage_ticks))) * (t if t > 0 else 1.0)
    if slip_points > 0:
        if side_ == "ENTRY":
            price = price + slip_points if dir_ == "LONG" else price - slip_points
        else:
            price = price - slip_points if dir_ == "LONG" else price + slip_points
    return float(price)


def best_worst_close_to_targets(direction: str, tp: float, sl: float, high: float, low: float) -> Tuple[float, float]:
    """Retorna:
    - preço mais próximo do TP atingido (no candle) e distância
    - preço mais próximo do SL atingido (no candle) e distância
    """
    if (direction or "").upper() == "LONG":
        # melhor a favor: high, pior contra: low
        closest_tp_price = min(high, tp)
        closest_sl_price = max(low, sl)
        return closest_tp_price, closest_sl_price
    # short
    closest_tp_price = max(low, tp)
    closest_sl_price = min(high, sl)
    return closest_tp_price, closest_sl_price


def run_backtest(
    candles_30s: List[Candle],
    analyzer_fn,
    *,
    symbol: str = "WINFUT",
    capital: float = 10_000.0,
    score_min: int = 70,
    allow_parallel: bool = True,
    max_parallel: int = 5,
    contracts: int = 1,
    point_value_brl: float = 0.2,
    tick_size_points: float = 5.0,
    slippage_ticks: int = 0,
    brokerage_brl_per_contract_per_side: float = 0.0,
    exchange_fee_brl_per_contract_per_side: float = 0.0,
    iss_pct_on_brokerage: float = 0.0,
) -> Dict[str, Any]:
    """Roda backtest bar-a-bar.

    analyzer_fn: função compatível com `scalping_analyzer.analyzer.get_full_analysis(candles, capital)`.
    """

    if not candles_30s or len(candles_30s) < 80:
        return {"error": "Arquivo insuficiente. Precisa pelo menos ~80 candles de 30s."}

    candles_30s = compute_indicators(candles_30s)

    open_trades: Dict[str, Dict[str, Any]] = {}
    closed_trades: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []

    def _log(kind: str, payload: Dict[str, Any]) -> None:
        logs.append({"ts_ms": payload.get("ts_ms"), "kind": kind, **payload})

    def _open_trade(direction: str, i: int, analysis: Dict[str, Any]) -> None:
        nonlocal open_trades
        c = candles_30s[i]
        trade_id = f"bt_{symbol}_{c.close_time}_{len(open_trades)+len(closed_trades)+1}"
        rp = ((analysis.get("score_analysis") or {}).get("risk_params") or {})
        # níveis (tp/sl) arredondados para tick (neutro) – execução (slippage/tick adverso) é aplicada no fill
        tp_raw = float(rp.get("take_profit_2") or rp.get("take_profit_1") or c.close)
        sl_raw = float(rp.get("stop_loss") or c.close)
        tp = _round_to_tick(tp_raw, tick_size_points, mode="nearest")
        sl = _round_to_tick(sl_raw, tick_size_points, mode="nearest")
        entry_raw = float(c.close)
        entry = _apply_execution_friction(
            price=entry_raw,
            direction=direction,
            side="ENTRY",
            tick_size_points=tick_size_points,
            slippage_ticks=slippage_ticks,
        )

        entry_ind = {
            "price": entry,
            "vwap": c.vwap,
            "ema9": c.ema9,
            "ema21": c.ema21,
            "ema55": c.ema55,
            "ema200": c.ema200,
            "rsi14": c.rsi,
            "atr14": c.atr,
            "macd": c.macd,
            "macd_signal": c.macd_signal,
            "macd_hist": c.macd_hist,
            "volume": c.volume,
        }

        open_trades[trade_id] = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "entry_time_ms": c.close_time,
            "entry_price": entry,
            "tp": tp,
            "sl": sl,
            "contracts": contracts,
            "point_value_brl": point_value_brl,
            "tick_size_points": tick_size_points,
            "slippage_ticks": slippage_ticks,
            "brokerage_brl_per_contract_per_side": brokerage_brl_per_contract_per_side,
            "exchange_fee_brl_per_contract_per_side": exchange_fee_brl_per_contract_per_side,
            "iss_pct_on_brokerage": iss_pct_on_brokerage,
            "entry_indicators": entry_ind,
            "entry_score": (analysis.get("score_analysis") or {}).get("score"),
            "entry_recommendation": (analysis.get("score_analysis") or {}).get("recommendation"),
            "samples": [],
            "closest_tp": {"dist": float("inf"), "price": None, "ts_ms": None},
            "closest_sl": {"dist": float("inf"), "price": None, "ts_ms": None},
            "status": "OPEN",
        }

        _log(
            "TRADE_OPEN",
            {
                "ts_ms": c.close_time,
                "trade_id": trade_id,
                "direction": direction,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "score": (analysis.get("score_analysis") or {}).get("score"),
            },
        )

    def _close_trade(tr: Dict[str, Any], exit_price: float, exit_ts_ms: int, reason: str) -> None:
        # aplica execução adversa no EXIT também
        exec_exit = _apply_execution_friction(
            price=float(exit_price),
            direction=str(tr.get("direction") or ""),
            side="EXIT",
            tick_size_points=float(tr.get("tick_size_points") or 0.0),
            slippage_ticks=int(tr.get("slippage_ticks") or 0),
        )
        tr["exit_price"] = float(exec_exit)
        tr["exit_time_ms"] = int(exit_ts_ms)
        tr["exit_reason"] = reason
        tr["result"] = "TP" if reason.startswith("TP") else "SL" if reason.startswith("SL") else "TIME"
        tr["status"] = "CLOSED"
        tr["gross_pnl_brl"] = calc_gross_pnl_b3(
            tr["entry_price"],
            tr["exit_price"],
            tr["direction"],
            tr["contracts"],
            tr["point_value_brl"],
        )
        tr["fees_brl"] = calc_fees_b3(
            contracts=int(tr.get("contracts") or 1),
            brokerage_brl_per_contract_per_side=float(tr.get("brokerage_brl_per_contract_per_side") or 0.0),
            exchange_fee_brl_per_contract_per_side=float(tr.get("exchange_fee_brl_per_contract_per_side") or 0.0),
            iss_pct_on_brokerage=float(tr.get("iss_pct_on_brokerage") or 0.0),
            sides=2,
        )
        tr["net_pnl_brl"] = float(tr["gross_pnl_brl"]) - float(tr["fees_brl"])
        closed_trades.append(tr)
        _log(
            "TRADE_CLOSE",
            {
                "ts_ms": exit_ts_ms,
                "trade_id": tr["trade_id"],
                "reason": reason,
                "exit": tr["exit_price"],
                "gross_pnl_brl": tr["gross_pnl_brl"],
                "fees_brl": tr["fees_brl"],
                "net_pnl_brl": tr["net_pnl_brl"],
            },
        )

    # loop
    lookback = 200
    for i in range(60, len(candles_30s)):
        window = candles_30s[max(0, i - lookback + 1) : i + 1]
        last = window[-1]

        # coleta samples 30s para trades abertos
        for tid, tr in list(open_trades.items()):
            # amostra
            tr["samples"].append(
                {
                    "ts_ms": last.close_time,
                    "price": last.close,
                    "high": last.high,
                    "low": last.low,
                    "vwap": last.vwap,
                    "ema9": last.ema9,
                    "ema21": last.ema21,
                    "ema55": last.ema55,
                    "ema200": last.ema200,
                    "rsi14": last.rsi,
                    "atr14": last.atr,
                    "macd": last.macd,
                    "macd_signal": last.macd_signal,
                    "macd_hist": last.macd_hist,
                    "volume": last.volume,
                }
            )

            # atualiza closest TP/SL
            tp = float(tr["tp"])
            sl = float(tr["sl"])
            dir_ = str(tr["direction"]).upper()
            if dir_ == "LONG":
                dist_tp = abs(tp - last.high)
                dist_sl = abs(last.low - sl)
                if dist_tp < tr["closest_tp"]["dist"]:
                    tr["closest_tp"].update({"dist": dist_tp, "price": last.high, "ts_ms": last.close_time})
                if dist_sl < tr["closest_sl"]["dist"]:
                    tr["closest_sl"].update({"dist": dist_sl, "price": last.low, "ts_ms": last.close_time})
            else:
                dist_tp = abs(last.low - tp)
                dist_sl = abs(sl - last.high)
                if dist_tp < tr["closest_tp"]["dist"]:
                    tr["closest_tp"].update({"dist": dist_tp, "price": last.low, "ts_ms": last.close_time})
                if dist_sl < tr["closest_sl"]["dist"]:
                    tr["closest_sl"].update({"dist": dist_sl, "price": last.high, "ts_ms": last.close_time})

            # verifica stop/target no candle (pior caso quando ambos atingem)
            hit_tp = False
            hit_sl = False
            if dir_ == "LONG":
                hit_sl = last.low <= sl
                hit_tp = last.high >= tp
            else:
                hit_sl = last.high >= sl
                hit_tp = last.low <= tp

            if hit_sl and hit_tp:
                # conservador: assume SL primeiro
                _close_trade(tr, sl, last.close_time, "SL (ambos no mesmo candle)")
                open_trades.pop(tid, None)
                continue
            if hit_sl:
                _close_trade(tr, sl, last.close_time, "SL")
                open_trades.pop(tid, None)
                continue
            if hit_tp:
                _close_trade(tr, tp, last.close_time, "TP")
                open_trades.pop(tid, None)
                continue

        # entrada (se permitido)
        if not allow_parallel and open_trades:
            continue
        if len(open_trades) >= max_parallel:
            continue

        analysis = analyzer_fn(window, capital=capital)
        if not isinstance(analysis, dict) or "error" in analysis:
            continue

        score = int(((analysis.get("score_analysis") or {}).get("score") or 0))
        direction_pt = str((analysis.get("score_analysis") or {}).get("direction") or "").upper()

        # direction já está traduzido para PT no analyzer, mas aceitamos ambas
        direction = "LONG" if direction_pt in ("COMPRA", "LONG", "BUY") else "SHORT" if direction_pt in ("VENDA", "SHORT", "SELL") else "NEUTRO"
        if direction == "NEUTRO":
            continue

        # valida confluência
        if direction == "LONG" and not bool((analysis.get("long_setup") or {}).get("is_ready")):
            continue
        if direction == "SHORT" and not bool((analysis.get("short_setup") or {}).get("is_ready")):
            continue

        if score >= score_min:
            _open_trade(direction, i, analysis)

    # fecha trades que sobraram no último preço
    if candles_30s:
        last = candles_30s[-1]
        for tid, tr in list(open_trades.items()):
            _close_trade(tr, float(last.close), int(last.close_time), "TIME")
            open_trades.pop(tid, None)

    # resumo
    tp = sum(1 for t in closed_trades if t.get("exit_reason", "").startswith("TP"))
    sl = sum(1 for t in closed_trades if t.get("exit_reason", "").startswith("SL"))
    time_closed = sum(1 for t in closed_trades if t.get("exit_reason") == "TIME")
    gross_pnl_brl = sum(float(t.get("gross_pnl_brl") or 0.0) for t in closed_trades)
    fees_brl = sum(float(t.get("fees_brl") or 0.0) for t in closed_trades)
    net_pnl_brl = sum(float(t.get("net_pnl_brl") or 0.0) for t in closed_trades)

    wins = [float(t.get("net_pnl_brl") or 0.0) for t in closed_trades if float(t.get("net_pnl_brl") or 0.0) > 0]
    losses = [float(t.get("net_pnl_brl") or 0.0) for t in closed_trades if float(t.get("net_pnl_brl") or 0.0) < 0]
    win_rate = (len(wins) / len(closed_trades)) if closed_trades else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else (float("inf") if avg_win else 0.0)

    # equity curve e max drawdown (sobre net)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    equity_curve: List[Dict[str, Any]] = []
    for t in sorted(closed_trades, key=lambda x: int(x.get("exit_time_ms") or 0)):
        equity += float(t.get("net_pnl_brl") or 0.0)
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
        equity_curve.append({"ts_ms": int(t.get("exit_time_ms") or 0), "equity_brl": equity})

    return {
        "symbol": symbol,
        "interval": "30s",
        "candles": len(candles_30s),
        "trades": len(closed_trades),
        "tp": tp,
        "sl": sl,
        "time": time_closed,
        "gross_pnl_brl": gross_pnl_brl,
        "fees_brl": fees_brl,
        "net_pnl_brl": net_pnl_brl,
        "win_rate": win_rate,
        "avg_win_brl": avg_win,
        "avg_loss_brl": avg_loss,
        "profit_factor": profit_factor,
        "payoff": payoff,
        "max_drawdown_brl": max_dd,
        "execution": {
            "contracts": contracts,
            "point_value_brl": point_value_brl,
            "tick_size_points": tick_size_points,
            "slippage_ticks": slippage_ticks,
            "brokerage_brl_per_contract_per_side": brokerage_brl_per_contract_per_side,
            "exchange_fee_brl_per_contract_per_side": exchange_fee_brl_per_contract_per_side,
            "iss_pct_on_brokerage": iss_pct_on_brokerage,
        },
        "equity_curve": equity_curve,
        "closed_trades": closed_trades,
        "logs": logs,
    }
