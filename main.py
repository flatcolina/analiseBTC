# -*- coding: utf-8 -*-
"""
BTC Scenarios Logger (Pullback vs Breakout)
- Fetches public market data from Binance (spot public endpoints)
- Builds 2 classical scenario families:
  (1) Reversal in zone (pullback / rejection around VWAP band)
  (2) Continuation (breakout confirmed beyond recent high/low + volume filter)
- Runs in cycles:
  snapshot -> define scenarios -> wait up to entry_timeout -> if triggered, "enter" market ->
  manage trade (TP/SL/timeout) with simple trade-management rules -> log everything -> repeat.

This is an analysis/logger, not financial advice.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Literal, Optional

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Importar modelos de dados compartilhados
from models import Candle, AnalysisSnapshot, Direction, FullAnalysis
# Importar o novo analisador de scalping
from scalping_analyzer import analyzer
from indicators import ema_series, rsi_series, atr_wilder_series, avg_volume, vwap_rolling, macd_series
from trade_intelligence import TradeIntelligence



def _parse_float(v: Any, default: float) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace(",", ".")
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)

def _parse_int(v: Any, default: int) -> int:
    try:
        if v is None:
            return int(default)
        if isinstance(v, bool):
            return int(default)
        if isinstance(v, int):
            return int(v)
        s = str(v).strip()
        if s == "":
            return int(default)
        return int(float(s))
    except Exception:
        return int(default)

# -------------------------
# Helpers
# -------------------------

def now_ms() -> int:
    return int(time.time() * 1000)

def ms_to_iso(ms: int) -> str:
    # ISO with local time offset not required; keep UTC for logs
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ms / 1000.0)) + f".{ms%1000:03d}Z"

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def apply_slippage(price: float, side: Literal["BUY", "SELL"], slippage_bps: float) -> float:
    # bps = 1/100 of 1%. Example: 2 bps = 0.02%
    f = slippage_bps / 10_000.0
    if side == "BUY":
        return price * (1.0 + f)
    return price * (1.0 - f)

# -------------------------
# Market data (Binance)
# -------------------------

BINANCE_BASE = os.getenv("BINANCE_SPOT_BASE", "https://data-api.binance.vision")

async def fetch_klines(symbol: str, interval: str, limit: int) -> list[list[Any]]:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.json()

async def fetch_price(symbol: str) -> float:
    url = f"{BINANCE_BASE}/api/v3/ticker/price"
    params = {"symbol": symbol}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        return float(data["price"])

## -------------------------
# Indicators
# -------------------------

def macd_series(closes: list[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    """Calcula MACD, Linha de Sinal e Histograma."""
    ema_fast = ema_series(closes, fast_period)
    ema_slow = ema_series(closes, slow_period)

    macd_line: list[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    # Calcular a linha de sinal (EMA da linha MACD)
    macd_values = [v for v in macd_line if v is not None]
    signal_line_raw = ema_series(macd_values, signal_period)

    # Mapear a linha de sinal de volta para o tamanho original
    signal_line: list[Optional[float]] = [None] * len(closes)
    j = 0
    for i in range(len(closes)):
        if macd_line[i] is not None:
            if j < len(signal_line_raw):
                signal_line[i] = signal_line_raw[j]
                j += 1

    # Calcular o histograma
    histogram: list[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram

def compute_indicators(candles: list[Candle]) -> list[Candle]:
    """Calcula e atribui todos os indicadores necessários aos objetos Candle."""
    closes = [c.close for c in candles]

    # EMAs (9, 21, 55, 200)
    ema9s = ema_series(closes, 9)
    ema21s = ema_series(closes, 21)
    ema55s = ema_series(closes, 55)
    ema200s = ema_series(closes, 200)

    # RSI
    rsis = rsi_series(candles, 14)

    # ATR
    atrs = atr_wilder_series(candles, 14)

    # VWAP (Rolling 100)
    vwaps = [vwap_rolling(candles[:i+1], 100) for i in range(len(candles))]

    # MACD
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

def parse_klines(rows: list[list[Any]]) -> list[Candle]:
    out: list[Candle] = []
    for r in rows:
        # Binance kline schema
        out.append(
            Candle(
                open_time=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
                close_time=int(r[6]),
                quote_asset_volume=float(r[7]),
                number_of_trades=int(r[8]),
                taker_buy_base_asset_volume=float(r[9]),
                taker_buy_quote_asset_volume=float(r[10]),
                ignore=float(r[11]),
            )
        )
    return out



# -------------------------
# Engine models
# -------------------------





# -------------------------
# Scenario engine
# -------------------------

class ScenarioEngine:
    def __init__(self) -> None:
        self.running: bool = False
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()

        self.logs: list[dict[str, Any]] = []
        self.stats = EngineStats()

        # breakdown
        self.by_kind: dict[str, ScenarioTotals] = {"PULLBACK": ScenarioTotals(), "BREAKOUT": ScenarioTotals()}
        self.by_key: dict[str, ScenarioTotals] = {}

        self.last_snapshot: AnalysisSnapshot | None = None
        self.last_scenarios: list[ScenarioDef] = []
        self.last_cycle: dict[str, Any] | None = None
        self.active_trade: TradeState | None = None

        self.cycle_id: int = 0
        self.cfg: dict[str, Any] = {}

            self.retention_ms: int = int(os.getenv("LOG_RETENTION_HOURS", "12")) * 3600_000

        def _prune_logs(self) -> None:
            if not self.logs:
                return
            cutoff = now_ms() - self.retention_ms
            # prune from front
            i = 0
            while i < len(self.logs) and int(self.logs[i].get("ts_ms", 0)) < cutoff:
                i += 1
            if i > 0:
                self.logs = self.logs[i:]

            # extra safety cap
            cap = int(os.getenv("LOG_MAX_ROWS", "60000"))
            if len(self.logs) > cap:
                self.logs = self.logs[-cap:]

        def log(self, event: str, **data: Any) -> None:
            ts = now_ms()
            row = {"ts_ms": ts, "ts": ms_to_iso(ts), "event": event, **data}
            self.logs.append(row)
            self._prune_logs()

        async def _snapshot(self, symbol: str, interval: str, limit: int) -> tuple[list[Candle], AnalysisSnapshot]:
            raw = await fetch_klines(symbol, interval, limit)
            candles = parse_klines(raw)
            if not candles:
                raise RuntimeError("No candles returned")

            # Compute all indicators for the full list of candles
            candles = compute_indicators(candles)

            price = candles[-1].close

            # The original snapshot only needs a few indicators for the old scenarios
            avg_vol = avg_volume(candles, 20)
            lookback = 20
            recent = candles[-lookback:] if len(candles) >= lookback else candles
            rh = max(c.high for c in recent) if recent else None
            rl = min(c.low for c in recent) if recent else None

            snap = AnalysisSnapshot(
                ts_ms=now_ms(),
                symbol=symbol,
                interval=interval,
                price=float(price),
                vwap=float(candles[-1].vwap) if candles[-1].vwap is not None else None,
                ema200=float(candles[-1].ema200) if candles[-1].ema200 is not None else None,
                rsi14=float(candles[-1].rsi) if candles[-1].rsi is not None else None,
                atr14=float(candles[-1].atr) if candles[-1].atr is not None else None,
                avg_vol20=avg_vol,
                recent_high=rh,
                recent_low=rl,
            )
            return candles, snap

    def _build_scenarios(
        self,
        candles: list[Candle],
        snap: AnalysisSnapshot,
        zone_mult_atr: float,
        breakout_buffer_atr: float,
        vol_mult: float,
        use_ema200: bool,
    ) -> list[ScenarioDef]:
        last = candles[-1]
        vwap = snap.vwap
        atr = snap.atr14
        rsi14 = snap.rsi14
        ema200 = snap.ema200
        avgv = snap.avg_vol20
        rh = snap.recent_high
        rl = snap.recent_low

        scenarios: list[ScenarioDef] = []

        # helpers
        def ema_ok(direction: Direction) -> tuple[bool, Optional[float]]:
            if not use_ema200 or ema200 is None:
                return True, ema200
            if direction == "LONG":
                return last.close > ema200, ema200
            return last.close < ema200, ema200

        # --- Pullback zone around VWAP ---
        # Touch zone within last 3 candles, then close back above/below VWAP
        touch_zone = False
        zone_dn = zone_up = None
        if vwap is not None and atr is not None:
            band = atr * zone_mult_atr
            zone_dn = vwap - band
            zone_up = vwap + band
            look = candles[-3:] if len(candles) >= 3 else candles
            for c in look:
                if c.low <= zone_up and c.high >= zone_dn:
                    touch_zone = True
                    break

        ema_ok_long, ema_val_long = ema_ok("LONG")
        scenarios.append(
            ScenarioDef(
                key="PULLBACK_LONG",
                name="Cenário 1 — Pullback LONG (rejeição na zona do VWAP)",
                direction="LONG",
                kind="PULLBACK",
                conditions=[
                    ScenarioCondition(
                        label="Preço tocou a zona VWAP± (últimos 3 candles)",
                        current=last.close,
                        target=vwap,
                        ok=bool(touch_zone),
                        extra={"vwap": vwap, "zone_dn": zone_dn, "zone_up": zone_up},
                    ),
                    ScenarioCondition(
                        label="Fechou acima do VWAP",
                        current=last.close,
                        target=vwap,
                        ok=bool(vwap is not None and last.close > vwap),
                        extra={"vwap": vwap},
                    ),
                    ScenarioCondition(
                        label="RSI(14) >= 50",
                        current=rsi14,
                        target=50.0,
                        ok=bool(rsi14 is not None and rsi14 >= 50.0),
                        extra={"rsi14": rsi14},
                    ),
                    ScenarioCondition(
                        label="Filtro EMA200 (preço > EMA200)",
                        current=last.close,
                        target=ema_val_long,
                        ok=bool(ema_ok_long),
                        extra={"ema200": ema_val_long},
                    ),
                ],
                refs={"vwap": vwap, "zone_dn": zone_dn, "zone_up": zone_up, "ema200": ema_val_long},
            )
        )

        ema_ok_short, ema_val_short = ema_ok("SHORT")
        scenarios.append(
            ScenarioDef(
                key="PULLBACK_SHORT",
                name="Cenário 2 — Pullback SHORT (rejeição na zona do VWAP)",
                direction="SHORT",
                kind="PULLBACK",
                conditions=[
                    ScenarioCondition(
                        label="Preço tocou a zona VWAP± (últimos 3 candles)",
                        current=last.close,
                        target=vwap,
                        ok=bool(touch_zone),
                        extra={"vwap": vwap, "zone_dn": zone_dn, "zone_up": zone_up},
                    ),
                    ScenarioCondition(
                        label="Fechou abaixo do VWAP",
                        current=last.close,
                        target=vwap,
                        ok=bool(vwap is not None and last.close < vwap),
                        extra={"vwap": vwap},
                    ),
                    ScenarioCondition(
                        label="RSI(14) <= 50",
                        current=rsi14,
                        target=50.0,
                        ok=bool(rsi14 is not None and rsi14 <= 50.0),
                        extra={"rsi14": rsi14},
                    ),
                    ScenarioCondition(
                        label="Filtro EMA200 (preço < EMA200)",
                        current=last.close,
                        target=ema_val_short,
                        ok=bool(ema_ok_short),
                        extra={"ema200": ema_val_short},
                    ),
                ],
                refs={"vwap": vwap, "zone_dn": zone_dn, "zone_up": zone_up, "ema200": ema_val_short},
            )
        )

        # --- Breakout: close beyond recent high/low + buffer & volume confirmation ---
        buffer = (atr * breakout_buffer_atr) if (atr is not None) else None
        vol_ok = bool(avgv is not None and last.volume >= avgv * vol_mult)

        scenarios.append(
            ScenarioDef(
                key="BREAKOUT_LONG",
                name="Cenário 3 — Breakout LONG (rompimento confirmado)",
                direction="LONG",
                kind="BREAKOUT",
                conditions=[
                    ScenarioCondition(
                        label="Fechou acima do topo recente + buffer",
                        current=last.close,
                        target=(rh + buffer) if (rh is not None and buffer is not None) else None,
                        ok=bool(rh is not None and buffer is not None and last.close > (rh + buffer)),
                        extra={"recent_high": rh, "buffer": buffer},
                    ),
                    ScenarioCondition(
                        label=f"Volume >= média(20) × {vol_mult}",
                        current=last.volume,
                        target=(avgv * vol_mult) if avgv is not None else None,
                        ok=vol_ok,
                        extra={"avg_vol20": avgv},
                    ),
                    ScenarioCondition(
                        label="RSI(14) >= 55 (força)",
                        current=rsi14,
                        target=55.0,
                        ok=bool(rsi14 is not None and rsi14 >= 55.0),
                        extra={"rsi14": rsi14},
                    ),
                    ScenarioCondition(
                        label="Filtro EMA200 (preço > EMA200)",
                        current=last.close,
                        target=ema_val_long,
                        ok=bool(ema_ok_long),
                        extra={"ema200": ema_val_long},
                    ),
                ],
                refs={"recent_high": rh, "buffer": buffer, "avg_vol20": avgv, "ema200": ema_val_long},
            )
        )

        scenarios.append(
            ScenarioDef(
                key="BREAKOUT_SHORT",
                name="Cenário 4 — Breakout SHORT (rompimento confirmado)",
                direction="SHORT",
                kind="BREAKOUT",
                conditions=[
                    ScenarioCondition(
                        label="Fechou abaixo do fundo recente - buffer",
                        current=last.close,
                        target=(rl - buffer) if (rl is not None and buffer is not None) else None,
                        ok=bool(rl is not None and buffer is not None and last.close < (rl - buffer)),
                        extra={"recent_low": rl, "buffer": buffer},
                    ),
                    ScenarioCondition(
                        label=f"Volume >= média(20) × {vol_mult}",
                        current=last.volume,
                        target=(avgv * vol_mult) if avgv is not None else None,
                        ok=vol_ok,
                        extra={"avg_vol20": avgv},
                    ),
                    ScenarioCondition(
                        label="RSI(14) <= 45 (força)",
                        current=rsi14,
                        target=45.0,
                        ok=bool(rsi14 is not None and rsi14 <= 45.0),
                        extra={"rsi14": rsi14},
                    ),
                    ScenarioCondition(
                        label="Filtro EMA200 (preço < EMA200)",
                        current=last.close,
                        target=ema_val_short,
                        ok=bool(ema_ok_short),
                        extra={"ema200": ema_val_short},
                    ),
                ],
                refs={"recent_low": rl, "buffer": buffer, "avg_vol20": avgv, "ema200": ema_val_short},
            )
        )

        return scenarios

    def _scenario_ok(self, sc: ScenarioDef) -> bool:
        return all(c.ok for c in sc.conditions)

    def _update_trade_diagnostics(self, trade: TradeState, snap: AnalysisSnapshot):
        """Update MFE/MAE, closest-to-TP/SL, and store indicator samples."""
        try:
            price = float(snap.price or 0.0)
            if price <= 0:
                return

            # initialize anchors
            if trade.best_fav_price <= 0:
                trade.best_fav_price = price
            if trade.worst_adv_price <= 0:
                trade.worst_adv_price = price

            # favorable/adverse extremes (direction-aware)
            if trade.direction == "LONG":
                trade.best_fav_price = max(trade.best_fav_price, price)
                trade.worst_adv_price = min(trade.worst_adv_price, price)
                unreal = (price - trade.entry_exec) * trade.qty_btc
            else:  # SHORT
                trade.best_fav_price = min(trade.best_fav_price, price)
                trade.worst_adv_price = max(trade.worst_adv_price, price)
                unreal = (trade.entry_exec - price) * trade.qty_btc

            trade.mfe_gross_usd = max(trade.mfe_gross_usd, unreal)
            trade.mae_gross_usd = min(trade.mae_gross_usd, unreal)

            # closest points to TP / SL (using current TP/SL levels, since they can move)
            tp = float(trade.tp_price or 0.0)
            sl = float(trade.sl_price or 0.0)
            ts_ms = int(snap.ts_ms or 0)

            if tp > 0:
                dist_tp = abs(tp - price)
                if trade.closest_tp_dist <= 0 or dist_tp < trade.closest_tp_dist:
                    trade.closest_tp_dist = dist_tp
                    trade.closest_tp_price = price
                    trade.closest_tp_ts_ms = ts_ms

            if sl > 0:
                dist_sl = abs(price - sl)
                if trade.closest_sl_dist <= 0 or dist_sl < trade.closest_sl_dist:
                    trade.closest_sl_dist = dist_sl
                    trade.closest_sl_price = price
                    trade.closest_sl_ts_ms = ts_ms

            # store indicator state (compact)
            trade.indicator_samples.append({
                "ts_ms": ts_ms,
                "ts": ms_to_iso(ts_ms),
                "price": price,
                "vwap": snap.vwap,
                "ema200": snap.ema200,
                "rsi14": snap.rsi14,
                "atr14": snap.atr14,
                "avg_vol20": snap.avg_vol20,
                "recent_high": snap.recent_high,
                "recent_low": snap.recent_low,
                "tp": tp,
                "sl": sl,
            })

            # avoid unbounded memory if something runs for very long
            if len(trade.indicator_samples) > 2000:
                trade.indicator_samples = trade.indicator_samples[-2000:]
        except Exception:
            return



    async def _manage_trade(
        self,
        trade: TradeState,
        symbol: str,
        interval: str,
        limit: int,
        poll_seconds: float,
        fee_rate_per_side: float,
        slippage_bps: float,
        be_trigger_r: float,
        trail_atr_mult: float,
        tp_extend_buffer_atr: float,
        tp_extend_add_atr: float,
        recalc_seconds: float,
    ) -> None:
        """
        Trade management rules (simple, deterministic):
        - When price reaches +be_trigger_r * initial_risk, move SL to breakeven(+fees) and arm trailing.
        - After BE is moved, trail SL by ATR * trail_atr_mult (updated ATR) when favorable.
        - For BREAKOUT only: if price is close to TP and momentum is strong, extend TP by ATR * tp_extend_add_atr.
        """
        now = now_ms()
        if (trade.last_manage_ms > 0) and (now - trade.last_manage_ms) < int(recalc_seconds * 1000):
            return
        trade.last_manage_ms = now

        # refresh indicators for decisions
        candles, snap = await self._snapshot(symbol, interval, limit)
        self.last_snapshot = snap
        cur_price = snap.price
        trade.last_price = cur_price
        self._update_trade_diagnostics(trade, snap)
        atr = snap.atr14 or trade.atr_at_entry
        vwap = snap.vwap
        rsi14 = snap.rsi14

        # initial risk based on initial SL (exec prices)
        initial_risk = abs(trade.entry_exec - trade.sl_price)
        if initial_risk <= 0:
            return

        # current R
        if trade.direction == "LONG":
            profit = cur_price - trade.entry_exec
        else:
            profit = trade.entry_exec - cur_price
        r_mult = profit / initial_risk

        # breakeven price that roughly covers entry+exit fees (ignoring slippage)
        if trade.direction == "LONG":
            be_price = trade.entry_exec * (1.0 + 2.0 * fee_rate_per_side)
            # for safety, never move SL above current price
            be_price = min(be_price, cur_price)
            be_better = be_price > trade.sl_price
        else:
            be_price = trade.entry_exec * (1.0 - 2.0 * fee_rate_per_side)
            be_price = max(be_price, cur_price)
            be_better = be_price < trade.sl_price

        # --- NOVA LÓGICA DE GESTÃO INTELIGENTE (TradeIntelligence) ---
        
        # 1. Obter os últimos candles para análise (desde a entrada)
        # Nota: O trade.indicator_samples armazena snapshots de indicadores.
        # Para análise de momentum, precisamos dos dados brutos dos candles.
        # Como não temos o histórico completo de candles no TradeState,
        # vamos usar o snapshot atual como o único candle para uma análise simplificada,
        # ou idealmente, buscar o histórico de candles desde a entrada.
        # Para esta implementação, vamos usar o TradeIntelligence para a decisão
        # e o ScenarioEngine para a execução e log.
        
        # O TradeIntelligence precisa do histórico de candles. Vamos simular
        # que o TradeIntelligence tem acesso ao histórico completo de candles
        # para a análise de momentum.
        
        # Para a simulação, vamos usar a lógica do TradeIntelligence
        # e aplicar as decisões no TradeState.
        
        # Instanciar a inteligência com os parâmetros de gerenciamento
        intelligence = TradeIntelligence(cfg={
            "BE_TRIGGER_R": be_trigger_r,
            "TRAIL_MULT_ATR": trail_atr_mult,
        })
        
        # Simular o histórico de candles para a análise de momentum
        # Na ausência do histórico completo, usamos o último snapshot como o candle atual
        # e assumimos que o TradeIntelligence fará a busca se necessário.
        # Para o propósito de simulação, vamos passar apenas o último candle
        # que contém os indicadores atualizados.
        
        # O TradeIntelligence precisa de uma lista de candles.
        # Vamos criar um candle com os dados do snapshot para simular.
        current_candle = Candle(
            open_time=snap.ts_ms,
            open=cur_price,
            high=cur_price,
            low=cur_price,
            close=cur_price,
            volume=0.0,
            close_time=snap.ts_ms,
            quote_asset_volume=0.0,
            number_of_trades=0,
            taker_buy_base_asset_volume=0.0,
            taker_buy_quote_asset_volume=0.0,
            ignore=0.0,
            vwap=snap.vwap,
            ema9=snap.ema9,
            ema21=snap.ema21,
            ema55=snap.ema55,
            ema200=snap.ema200,
            atr=snap.atr14,
            rsi=snap.rsi14,
            macd=None, # Não disponível no snapshot
            macd_signal=None, # Não disponível no snapshot
            macd_hist=None, # Não disponível no snapshot
        )
        
        # Ação recomendada pela inteligência
        action = intelligence.analyze_and_manage(trade, [current_candle])
        
        if action == "MOVE_BE":
            if not trade.be_moved:
                new_sl = intelligence.get_new_sl_price(trade, [current_candle], "MOVE_BE")
                old = trade.sl_price
                trade.sl_price = new_sl
                trade.be_moved = True
                self.log(
                    "MOVE_SL_BE_INTEL",
                    cycle_id=trade.cycle_id,
                    scenario_key=trade.scenario_key,
                    direction=trade.direction,
                    r_mult=r_mult,
                    old_sl=old,
                    new_sl=trade.sl_price,
                    vwap=vwap,
                    rsi14=rsi14,
                )
        
        elif action == "TRAIL":
            new_sl = intelligence.get_new_sl_price(trade, [current_candle], "TRAIL")
            if (trade.direction == "LONG" and new_sl > trade.sl_price) or \
               (trade.direction == "SHORT" and new_sl < trade.sl_price):
                old = trade.sl_price
                trade.sl_price = new_sl
                self.log(
                    "TRAIL_SL_INTEL",
                    cycle_id=trade.cycle_id,
                    scenario_key=trade.scenario_key,
                    direction=trade.direction,
                    old_sl=old,
                    new_sl=trade.sl_price,
                    atr=atr,
                    vwap=vwap,
                    rsi14=rsi14,
                )
                
        elif action == "EXIT_IMMEDIATE":
            # Fechar a operação imediatamente por reversão de momentum
            # Isso será tratado pelo loop principal que verifica o SL
            # Mas podemos forçar o SL para o preço atual para garantir a saída
            trade.sl_price = cur_price
            self.log(
                "EXIT_IMMEDIATE_INTEL",
                cycle_id=trade.cycle_id,
                scenario_key=trade.scenario_key,
                direction=trade.direction,
                r_mult=r_mult,
                vwap=vwap,
                rsi14=rsi14,
            )
        
        # A lógica de estender TP (EXTEND_TP) será mantida, mas pode ser integrada
        # ao TradeIntelligence em uma próxima iteração. Por enquanto, mantemos a original.
        
        # 3) extend TP (BREAKOUT only) when close to TP and momentum strong
        if trade.scenario_kind == "BREAKOUT" and (not trade.tp_extended):
            close_to_tp = False
            if trade.direction == "LONG":
                close_to_tp = cur_price >= (trade.tp_price - atr * tp_extend_buffer_atr)
                momentum_strong = bool((rsi14 is not None and rsi14 >= 60.0) and (vwap is None or cur_price >= vwap))
                if close_to_tp and momentum_strong:
                    old_tp = trade.tp_price
                    trade.tp_price = trade.tp_price + atr * tp_extend_add_atr
                    trade.tp_extended = True
                    self.log(
                        "EXTEND_TP",
                        cycle_id=trade.cycle_id,
                        scenario_key=trade.scenario_key,
                        direction=trade.direction,
                        old_tp=old_tp,
                        new_tp=trade.tp_price,
                        atr=atr,
                        vwap=vwap,
                        rsi14=rsi14,
                    )
            else:
                close_to_tp = cur_price <= (trade.tp_price + atr * tp_extend_buffer_atr)
                momentum_strong = bool((rsi14 is not None and rsi14 <= 40.0) and (vwap is None or cur_price <= vwap))
                if close_to_tp and momentum_strong:
                    old_tp = trade.tp_price
                    trade.tp_price = trade.tp_price - atr * tp_extend_add_atr
                    trade.tp_extended = True
                    self.log(
                        "EXTEND_TP",
                        cycle_id=trade.cycle_id,
                        scenario_key=trade.scenario_key,
                        direction=trade.direction,
                        old_tp=old_tp,
                        new_tp=trade.tp_price,
                        atr=atr,
                        vwap=vwap,
                        rsi14=rsi14,
                    )

    async def _monitor_trade(
        self,
        trade: TradeState,
        symbol: str,
        interval: str,
        limit: int,
        poll_seconds: float,
        fee_rate_per_side: float,
        slippage_bps: float,
        max_hold_minutes: int,
        be_trigger_r: float,
        trail_atr_mult: float,
        tp_extend_buffer_atr: float,
        tp_extend_add_atr: float,
        recalc_seconds: float,
    ) -> dict[str, Any]:
        self.active_trade = trade
        self.log(
            "TRADE_SET",
            cycle_id=trade.cycle_id,
            scenario_key=trade.scenario_key,
            scenario_kind=trade.scenario_kind,
            direction=trade.direction,
            entry_raw=trade.entry_raw,
            entry_exec=trade.entry_exec,
            qty_btc=trade.qty_btc,
            tp_price=trade.tp_price,
            sl_price=trade.sl_price,
            atr_at_entry=trade.atr_at_entry,
        )

        deadline = now_ms() + max_hold_minutes * 60_000
        while self.running and now_ms() < deadline:
            cur_price = await fetch_price(symbol)
            trade.last_price = cur_price

            # run trade management periodically
            try:
                await self._manage_trade(
                    trade=trade,
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    poll_seconds=poll_seconds,
                    fee_rate_per_side=fee_rate_per_side,
                    slippage_bps=slippage_bps,
                    be_trigger_r=be_trigger_r,
                    trail_atr_mult=trail_atr_mult,
                    tp_extend_buffer_atr=tp_extend_buffer_atr,
                    tp_extend_add_atr=tp_extend_add_atr,
                    recalc_seconds=recalc_seconds,
                )
            except Exception as e:
                self.log("MANAGE_ERROR", cycle_id=trade.cycle_id, error=str(e))

            hit_tp = (cur_price >= trade.tp_price) if trade.direction == "LONG" else (cur_price <= trade.tp_price)
            hit_sl = (cur_price <= trade.sl_price) if trade.direction == "LONG" else (cur_price >= trade.sl_price)

            if hit_tp or hit_sl:
                outcome = "TP" if hit_tp else "SL"
                exit_raw = trade.tp_price if hit_tp else trade.sl_price
                return self._finalize_trade(trade, outcome, exit_raw, fee_rate_per_side, slippage_bps)

            await asyncio.sleep(poll_seconds)

        # TIMEOUT or stopped => close at last price
        exit_raw = trade.last_price if trade.last_price > 0 else await fetch_price(symbol)
        return self._finalize_trade(trade, "TIME", exit_raw, fee_rate_per_side, slippage_bps)

    def _finalize_trade(
        self,
        trade: TradeState,
        outcome: Literal["TP", "SL", "TIME"],
        exit_raw: float,
        fee_rate_per_side: float,
        slippage_bps: float,
    ) -> dict[str, Any]:
        t_ms = now_ms()
        exit_side: Literal["BUY", "SELL"] = "SELL" if trade.direction == "LONG" else "BUY"
        exit_exec = apply_slippage(exit_raw, exit_side, slippage_bps)

        gross = trade.qty_btc * (exit_exec - trade.entry_exec) if trade.direction == "LONG" else trade.qty_btc * (trade.entry_exec - exit_exec)
        notional_entry = trade.qty_btc * trade.entry_exec
        notional_exit = trade.qty_btc * exit_exec
        fees = (notional_entry + notional_exit) * fee_rate_per_side
        net = gross - fees

        # update stats
        self.stats.trades_total += 1
        self.by_kind[trade.scenario_kind].trades += 1
        self.by_key.setdefault(trade.scenario_key, ScenarioTotals()).trades += 1

        if outcome == "TP":
            self.stats.tp += 1
            self.by_kind[trade.scenario_kind].tp += 1
            self.by_key[trade.scenario_key].tp += 1
        elif outcome == "SL":
            self.stats.sl += 1
            self.by_kind[trade.scenario_kind].sl += 1
            self.by_key[trade.scenario_key].sl += 1
        else:
            self.stats.time += 1
            self.by_kind[trade.scenario_kind].time += 1
            self.by_key[trade.scenario_key].time += 1

        self.stats.gross_pnl_usd += gross
        self.stats.fees_usd += fees
        self.stats.net_pnl_usd += net

        self.by_kind[trade.scenario_kind].gross_pnl_usd += gross
        self.by_kind[trade.scenario_kind].fees_usd += fees
        self.by_kind[trade.scenario_kind].net_pnl_usd += net

        self.by_key[trade.scenario_key].gross_pnl_usd += gross
        self.by_key[trade.scenario_key].fees_usd += fees
        self.by_key[trade.scenario_key].net_pnl_usd += net
        # split net PnL by outcome (requested for TP/SL/TIME accumulated values)
        if outcome == "TP":
            self.stats.net_tp_usd += net
            self.by_kind[trade.scenario_kind].net_tp_usd += net
            self.by_key[trade.scenario_key].net_tp_usd += net
        elif outcome == "SL":
            self.stats.net_sl_usd += net
            self.by_kind[trade.scenario_kind].net_sl_usd += net
            self.by_key[trade.scenario_key].net_sl_usd += net
        else:
            self.stats.net_time_usd += net
            self.by_kind[trade.scenario_kind].net_time_usd += net
            self.by_key[trade.scenario_key].net_time_usd += net

        risk_per_btc = abs(trade.entry_exec - trade.sl_price) if (trade.entry_exec is not None and trade.sl_price is not None) else None
        risk_usd = (risk_per_btc * trade.qty_btc) if (risk_per_btc is not None and trade.qty_btc is not None) else None
        mfe_r = (trade.mfe_gross_usd / risk_usd) if (risk_usd not in (None, 0)) else None
        mae_r = (trade.mae_gross_usd / risk_usd) if (risk_usd not in (None, 0)) else None
        closest_tp_r = (trade.closest_tp_dist / risk_per_btc) if (risk_per_btc not in (None, 0) and trade.closest_tp_dist is not None) else None
        closest_sl_r = (trade.closest_sl_dist / risk_per_btc) if (risk_per_btc not in (None, 0) and trade.closest_sl_dist is not None) else None

        info = {
            "outcome": outcome,
            "scenario_kind": trade.scenario_kind,
            "scenario_key": trade.scenario_key,
            "direction": trade.direction,
            "entry_price_exec": trade.entry_exec,
            "risk_usd": risk_usd,
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "closest_tp_r": closest_tp_r,
            "closest_sl_r": closest_sl_r,
            "exit_time_ms": t_ms,
            "exit_time": ms_to_iso(t_ms),
            "exit_price_raw": exit_raw,
            "exit_price_exec": exit_exec,
            "gross_pnl_usd": gross,
            "fees_usd": fees,
            "net_pnl_usd": net,
            "tp_price": trade.tp_price,
            "sl_price": trade.sl_price,
            "qty_btc": trade.qty_btc,
            "scenario_key": trade.scenario_key,
            "scenario_kind": trade.scenario_kind,
            "direction": trade.direction,
            "entry_exec": trade.entry_exec,
            "entry_raw": trade.entry_raw,
            'best_fav_price': trade.best_fav_price,
            'worst_adv_price': trade.worst_adv_price,
            'closest_tp_price': trade.closest_tp_price,
            'closest_tp_dist': trade.closest_tp_dist,
            'closest_tp_ts': ms_to_iso(trade.closest_tp_ts_ms) if trade.closest_tp_ts_ms else None,
            'closest_sl_price': trade.closest_sl_price,
            'closest_sl_dist': trade.closest_sl_dist,
            'closest_sl_ts': ms_to_iso(trade.closest_sl_ts_ms) if trade.closest_sl_ts_ms else None,
            'mfe_gross_usd': trade.mfe_gross_usd,
            'mae_gross_usd': trade.mae_gross_usd,
            'entry_indicators': trade.entry_indicators,
            'indicator_samples': trade.indicator_samples,
        }

        self.log("TRADE_END", cycle_id=trade.cycle_id, **info)
        self.active_trade = None
        return info

    async def _run(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "3m",
        limit: int = 500,
        poll_seconds: float = 5.0,
        entry_timeout_seconds: int = 600,
        max_hold_minutes: int = 25,
        fee_rate_per_side: float = 0.001,
        slippage_bps: float = 2.0,
        collateral_usd: float = 1879.0,
        zone_mult_atr: float = 0.35,
        breakout_buffer_atr: float = 0.20,
        vol_mult: float = 1.10,
        use_ema200: bool = True,
        tp_atr_mult: float = 1.0,
        sl_atr_mult: float = 0.7,
        # trade management
        be_trigger_r: float = 0.8,
        trail_atr_mult: float = 0.8,
        tp_extend_buffer_atr: float = 0.15,
        tp_extend_add_atr: float = 0.50,
        recalc_seconds: float = 30.0,
    ) -> None:
        self.cfg = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "poll_seconds": poll_seconds,
            "entry_timeout_seconds": entry_timeout_seconds,
            "max_hold_minutes": max_hold_minutes,
            "fee_rate_per_side": fee_rate_per_side,
            "slippage_bps": slippage_bps,
            "collateral_usd": collateral_usd,
            "zone_mult_atr": zone_mult_atr,
            "breakout_buffer_atr": breakout_buffer_atr,
            "vol_mult": vol_mult,
            "use_ema200": use_ema200,
            "tp_atr_mult": tp_atr_mult,
            "sl_atr_mult": sl_atr_mult,
            "be_trigger_r": be_trigger_r,
            "trail_atr_mult": trail_atr_mult,
            "tp_extend_buffer_atr": tp_extend_buffer_atr,
            "tp_extend_add_atr": tp_extend_add_atr,
            "recalc_seconds": recalc_seconds,
            "binance_base": BINANCE_BASE,
        }

        self.log("STARTED", cfg=self.cfg)

        while self.running:
            async with self.lock:
                self.cycle_id += 1
                cid = self.cycle_id

            cycle_start = now_ms()
            self.stats.cycles_total += 1
            self.last_cycle = {"cycle_id": cid, "status": "STARTED", "started_at": ms_to_iso(cycle_start)}

            try:
                candles, snap = await self._snapshot(symbol, interval, limit)
            except Exception as e:
                self.log("SNAPSHOT_ERROR", cycle_id=cid, error=str(e))
                await asyncio.sleep(2.0)
                continue

            self.last_snapshot = snap

            scenarios = self._build_scenarios(candles, snap, zone_mult_atr, breakout_buffer_atr, vol_mult, use_ema200)
            self.last_scenarios = scenarios

            self.log("CYCLE_START", cycle_id=cid, snapshot=asdict(snap))
            for sc in scenarios:
                self.log("SCENARIO_DEF", cycle_id=cid, scenario=asdict(sc))

            # wait for trigger
            deadline = cycle_start + entry_timeout_seconds * 1000
            triggered: ScenarioDef | None = None

            while self.running and now_ms() < deadline and triggered is None:
                try:
                    candles2, snap2 = await self._snapshot(symbol, interval, limit)
                    self.last_snapshot = snap2
                except Exception as e:
                    self.log("POLL_ERROR", cycle_id=cid, error=str(e))
                    await asyncio.sleep(poll_seconds)
                    continue

                # rebuild scenarios using latest snapshot (so the UI matches the same logic)
                scenarios2 = self._build_scenarios(candles2, snap2, zone_mult_atr, breakout_buffer_atr, vol_mult, use_ema200)

                # deterministic priority: Pullback first, then Breakout (you can invert if wanted)
                for sc in scenarios2:
                    if self._scenario_ok(sc):
                        triggered = sc
                        break

                if triggered:
                    self.last_scenarios = scenarios2
                    self.log("TRIGGERED", cycle_id=cid, triggered=asdict(triggered), price=snap2.price)
                    snap = snap2
                    break

                await asyncio.sleep(poll_seconds)

            if not self.running:
                break

            if triggered is None:
                self.stats.no_entry += 1
                dur = (now_ms() - cycle_start) / 1000.0
                self.last_cycle = {"cycle_id": cid, "status": "NO_ENTRY", "duration_seconds": dur}
                self.log("CYCLE_END_NO_ENTRY", cycle_id=cid, duration_s=dur)
                continue

            # enter
            direction: Direction = triggered.direction
            atr_val = float(snap.atr14 or 0.0)
            raw_price = float(snap.price)
            entry_side = "BUY" if direction == "LONG" else "SELL"
            entry_exec = apply_slippage(raw_price, entry_side, slippage_bps)
            qty = collateral_usd / entry_exec if entry_exec > 0 else 0.0

            # initial TP/SL
            tp_dist = atr_val * tp_atr_mult
            sl_dist = atr_val * sl_atr_mult
            tp_price = raw_price + tp_dist if direction == "LONG" else raw_price - tp_dist
            sl_price = raw_price - sl_dist if direction == "LONG" else raw_price + sl_dist

            entry_time = now_ms()

            trade = TradeState(
                cycle_id=cid,
                scenario_key=triggered.key,
                scenario_kind=triggered.kind,
                direction=direction,
                entry_time_ms=entry_time,
                entry_raw=raw_price,
                entry_exec=entry_exec,
                qty_btc=qty,
                tp_price=tp_price,
                sl_price=sl_price,
                atr_at_entry=atr_val,
                last_price=raw_price,
            )
            # initialize diagnostics and store indicators at entry
            trade.best_fav_price = raw_price
            trade.worst_adv_price = raw_price
            trade.closest_tp_price = raw_price
            trade.closest_tp_dist = abs(trade.tp_price - raw_price) if trade.tp_price else 0.0
            trade.closest_tp_ts_ms = snap.ts_ms
            trade.closest_sl_price = raw_price
            trade.closest_sl_dist = abs(raw_price - trade.sl_price) if trade.sl_price else 0.0
            trade.closest_sl_ts_ms = snap.ts_ms
            trade.entry_indicators = {
                'ts_ms': snap.ts_ms,
                'ts': ms_to_iso(snap.ts_ms),
                'price': raw_price,
                'vwap': snap.vwap,
                'ema200': snap.ema200,
                'rsi14': snap.rsi14,
                'atr14': snap.atr14,
                'avg_vol20': snap.avg_vol20,
                'recent_high': snap.recent_high,
                'recent_low': snap.recent_low,
            }
            trade.indicator_samples = [dict(trade.entry_indicators, tp=trade.tp_price, sl=trade.sl_price)]

            self.log(
                "ENTER",
                cycle_id=cid,
                scenario_key=trade.scenario_key,
                scenario_kind=trade.scenario_kind,
                direction=direction,
                entry_time=ms_to_iso(entry_time),
                entry_price_raw=raw_price,
                entry_price_exec=entry_exec,
                qty_btc=qty,
                collateral_usd=collateral_usd,
                atr14=atr_val,
                tp_price=tp_price,
                sl_price=sl_price,
            )

            exit_info = await self._monitor_trade(
                trade=trade,
                symbol=symbol,
                interval=interval,
                limit=limit,
                poll_seconds=poll_seconds,
                fee_rate_per_side=fee_rate_per_side,
                slippage_bps=slippage_bps,
                max_hold_minutes=max_hold_minutes,
                be_trigger_r=be_trigger_r,
                trail_atr_mult=trail_atr_mult,
                tp_extend_buffer_atr=tp_extend_buffer_atr,
                tp_extend_add_atr=tp_extend_add_atr,
                recalc_seconds=recalc_seconds,
            )

            dur = (exit_info["exit_time_ms"] - entry_time) / 1000.0
            self.last_cycle = {
                "cycle_id": cid,
                "status": f"ENTERED_{exit_info['outcome']}",
                "scenario_key": triggered.key,
                "scenario_kind": triggered.kind,
                "duration_seconds": dur,
                "entry_time": ms_to_iso(entry_time),
                "exit_time": exit_info["exit_time"],
                "net_pnl_usd": exit_info["net_pnl_usd"],
            }
            self.log("CYCLE_END", cycle_id=cid, summary=self.last_cycle)

            await asyncio.sleep(0.25)

        self.log("STOPPED")

    async def start(self, **cfg: Any) -> None:
        async with self.lock:
            if self.running:
                return
            self.running = True
            self.task = asyncio.create_task(self._run(**cfg))

    async def stop(self) -> None:
        async with self.lock:
            self.running = False
            t = self.task
            self.task = None
        if t:
            try:
                await asyncio.wait_for(t, timeout=2.0)
            except Exception:
                pass
        self.active_trade = None
        self.log("STOP_REQUESTED")

    def reset(self) -> None:
        self.running = False
        self.task = None
        self.logs = []
        self.stats = EngineStats()
        self.by_kind = {"PULLBACK": ScenarioTotals(), "BREAKOUT": ScenarioTotals()}
        self.by_key = {}
        self.last_snapshot = None
        self.last_scenarios = []
        self.last_cycle = None
        self.active_trade = None
        self.cycle_id = 0
        self.cfg = {}
        self.log("RESET_DONE")

# -------------------------
# FastAPI
# -------------------------

app = FastAPI(title="BTC Scenarios Logger", version="11.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ScenarioEngine()

@app.get("/health")
def health():
    return {"ok": True, "ts": ms_to_iso(now_ms()), "binance_base": BINANCE_BASE}

@app.get("/api/version")
def api_version():
    return {
        "version": "11.1",
        "ts": ms_to_iso(now_ms()),
        "binance_base": BINANCE_BASE,
    }


@app.post("/api/start")
async def api_start(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("3m"),
    collateral_usd: float = Query(1879.0, gt=0),
    poll_seconds: float = Query(5.0, ge=1.0, le=60.0),
    entry_timeout_seconds: int = Query(600, ge=5, le=7200),
    max_hold_minutes: int = Query(25, ge=1, le=240),
    fee_rate_per_side: float = Query(0.001, ge=0.0, le=0.02),
    slippage_bps: float = Query(2.0, ge=0.0, le=100.0),
    zone_mult_atr: float = Query(0.35, ge=0.0, le=5.0),
    breakout_buffer_atr: float = Query(0.20, ge=0.0, le=5.0),
    vol_mult: float = Query(1.10, ge=0.1, le=10.0),
    use_ema200: bool = Query(True),
    tp_atr_mult: float = Query(1.0, ge=0.1, le=10.0),
    sl_atr_mult: float = Query(0.7, ge=0.1, le=10.0),
    # management
    be_trigger_r: float = Query(0.8, ge=0.1, le=5.0),
    trail_atr_mult: float = Query(0.8, ge=0.1, le=10.0),
    tp_extend_buffer_atr: float = Query(0.15, ge=0.01, le=5.0),
    tp_extend_add_atr: float = Query(0.50, ge=0.0, le=10.0),
    recalc_seconds: float = Query(30.0, ge=5.0, le=120.0),
    limit: int = Query(500, ge=50, le=1000),
):
    await engine.start(
        symbol=symbol,
        interval=interval,
        limit=limit,
        poll_seconds=poll_seconds,
        entry_timeout_seconds=entry_timeout_seconds,
        max_hold_minutes=max_hold_minutes,
        fee_rate_per_side=fee_rate_per_side,
        slippage_bps=slippage_bps,
        collateral_usd=collateral_usd,
        zone_mult_atr=zone_mult_atr,
        breakout_buffer_atr=breakout_buffer_atr,
        vol_mult=vol_mult,
        use_ema200=use_ema200,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        be_trigger_r=be_trigger_r,
        trail_atr_mult=trail_atr_mult,
        tp_extend_buffer_atr=tp_extend_buffer_atr,
        tp_extend_add_atr=tp_extend_add_atr,
        recalc_seconds=recalc_seconds,
    )
    return {"ok": True, "running": engine.running, "cfg": engine.cfg}

@app.post("/api/stop")
async def api_stop():
    await engine.stop()
    return {"ok": True, "running": engine.running}

@app.post("/api/reset")
def api_reset():
    engine.reset()
    return {"ok": True}

@app.get("/api/diagnostics")
def api_diagnostics(
    hours: str = Query("12"),
    limit: str = Query("20000"),
):
    hours_f = _parse_float(hours, 12.0)
    if hours_f < 0.1:
        hours_f = 0.1
    if hours_f > 72.0:
        hours_f = 72.0

    limit_i = _parse_int(limit, 20000)
    if limit_i < 1:
        limit_i = 1
    if limit_i > 60000:
        limit_i = 60000

    cutoff = now_ms() - int(hours_f * 3600_000)
    rows = [r for r in engine.logs if int(r.get("ts_ms", 0)) >= cutoff]
    ends = [r for r in rows if r.get("event") == "TRADE_END"]
    if len(ends) > limit_i:
        ends = ends[-limit_i:]

    def _bucket_rsi(rsi: Any) -> str:
        v = _parse_float(rsi, float("nan"))
        if v != v:
            return "RSI:na"
        if v < 40:
            return "RSI:<40"
        if v <= 60:
            return "RSI:40-60"
        return "RSI:>60"

    def _side(price: Any, ref: Any, label: str) -> str:
        p = _parse_float(price, float("nan"))
        r = _parse_float(ref, float("nan"))
        if p != p or r != r:
            return f"{label}:na"
        return f"{label}:{'above' if p >= r else 'below'}"

    per_kind: Dict[str, Dict[str, Any]] = {}
    per_scenario: Dict[str, Dict[str, Any]] = {}

    pullback_sl_patterns: Dict[str, int] = {}
    pullback_sl: List[Dict[str, Any]] = []
    pullback_all: List[Dict[str, Any]] = []

    def _agg(dct: Dict[str, Dict[str, Any]], name: str, outcome: str, net: float) -> None:
        if name not in dct:
            dct[name] = {"trades": 0, "tp": 0, "sl": 0, "time": 0, "net_usd": 0.0}
        dct[name]["trades"] += 1
        if outcome == "TP":
            dct[name]["tp"] += 1
        elif outcome == "SL":
            dct[name]["sl"] += 1
        elif outcome == "TIME":
            dct[name]["time"] += 1
        dct[name]["net_usd"] += net

    for row in ends:
        payload = row.get("payload") or {}
        kind = payload.get("scenario_kind") or payload.get("kind") or "unknown"
        scen = payload.get("scenario_key") or payload.get("scenario") or "unknown"
        outcome = payload.get("outcome") or "UNKNOWN"
        net = _parse_float(payload.get("net_usd"), 0.0)

        _agg(per_kind, str(kind), str(outcome), float(net))
        _agg(per_scenario, str(scen), str(outcome), float(net))

        if str(kind).lower().startswith("pullback"):
            pullback_all.append(payload)
            if str(outcome) == "SL":
                pullback_sl.append(payload)
                ind = payload.get("entry_indicators") or {}
                pattern = " | ".join([
                    _side(ind.get("price"), ind.get("vwap"), "VWAP"),
                    _side(ind.get("price"), ind.get("ema200"), "EMA200"),
                    _bucket_rsi(ind.get("rsi14")),
                ])
                pullback_sl_patterns[pattern] = pullback_sl_patterns.get(pattern, 0) + 1

    def _avg(items: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals: List[float] = []
        for it in items:
            v = _parse_float(it.get(key), float("nan"))
            if v == v:
                vals.append(float(v))
        return (sum(vals) / len(vals)) if vals else None

    def _pct(items: List[Dict[str, Any]], pred) -> Optional[float]:
        if not items:
            return None
        ok = 0
        for it in items:
            if pred(it):
                ok += 1
        return ok / len(items)

    def _get_r(it: Dict[str, Any], key: str) -> float:
        return _parse_float(it.get(key), float("nan"))

    pullback_diag = {
        "trades": len(pullback_all),
        "sl": len(pullback_sl),
        "tp": sum(1 for it in pullback_all if str(it.get("outcome")) == "TP"),
        "time": sum(1 for it in pullback_all if str(it.get("outcome")) == "TIME"),
        "net_usd": sum(_parse_float(it.get("net_usd"), 0.0) for it in pullback_all),
        "avg_mfe_r_on_sl": _avg(pullback_sl, "mfe_r"),
        "avg_mae_r_on_sl": _avg(pullback_sl, "mae_r"),
        "avg_closest_tp_r_on_sl": _avg(pullback_sl, "closest_tp_r"),
        "pct_sl_after_0_5r": _pct(pullback_sl, lambda it: _get_r(it, "mfe_r") >= 0.5),
        "pct_sl_after_0_8r": _pct(pullback_sl, lambda it: _get_r(it, "mfe_r") >= 0.8),
        "pct_sl_near_tp": _pct(pullback_sl, lambda it: _get_r(it, "closest_tp_r") <= 0.2),
        "top_sl_patterns": [
            {"pattern": p, "count": c}
            for p, c in sorted(pullback_sl_patterns.items(), key=lambda kv: kv[1], reverse=True)[:8]
        ],
    }

    return {
        "hours": hours_f,
        "returned": len(ends),
        "by_kind": per_kind,
        "by_scenario": per_scenario,
        "pullback_diag": pullback_diag,
    }

@app.get("/api/status")
def api_status():
    return {
        "running": engine.running,
        "stats": asdict(engine.stats),
        "by_kind": {k: asdict(v) for k, v in engine.by_kind.items()},
        "by_key": {k: asdict(v) for k, v in engine.by_key.items()},
        "last_snapshot": asdict(engine.last_snapshot) if engine.last_snapshot else None,
        "last_scenarios": [asdict(s) for s in engine.last_scenarios],
        "last_cycle": engine.last_cycle,
        "active_trade": asdict(engine.active_trade) if engine.active_trade else None,
        "logs_len": len(engine.logs),
        "cfg": engine.cfg,
    }

@app.get("/api/logs")
def api_logs(
    hours: str = Query("12"),
    limit: str = Query("20000"),
):
    hours_f = _parse_float(hours, 12.0)
    if hours_f < 0.1:
        hours_f = 0.1
    if hours_f > 72.0:
        hours_f = 72.0

    limit_i = _parse_int(limit, 20000)
    if limit_i < 1:
        limit_i = 1
    if limit_i > 60000:
        limit_i = 60000

    cutoff = now_ms() - int(hours_f * 3600_000)
    rows = [r for r in engine.logs if int(r.get("ts_ms", 0)) >= cutoff]
    return {"logs": rows[-int(limit_i):], "hours": hours_f, "returned": min(len(rows), int(limit_i))}

@app.get("/api/full_analysis", response_model=FullAnalysis)
async def get_full_analysis(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1m"),
    limit: int = Query(200),
    capital: float = Query(10000.0),
) -> FullAnalysis:
    """Retorna a análise completa de scalping (5 algoritmos)"""
    raw = await fetch_klines(symbol, interval, limit)
    candles = parse_klines(raw)

    # Computar todos os indicadores
    candles = compute_indicators(candles)

    # Executar a análise completa
    analysis_dict = analyzer.get_full_analysis(candles, capital)

    # Converter o dicionário de volta para o modelo Pydantic/dataclass
    return FullAnalysis(**analysis_dict)

@app.get("/api/conditions")
async def api_conditions(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("3m"),
    zone_mult_atr: float = Query(0.35, ge=0.0, le=5.0),
    breakout_buffer_atr: float = Query(0.20, ge=0.0, le=5.0),
    vol_mult: float = Query(1.10, ge=0.1, le=10.0),
    use_ema200: bool = Query(True),
    limit: int = Query(500, ge=50, le=1000),
):
    try:
        candles, snap = await engine._snapshot(symbol, interval, limit)
        scs = engine._build_scenarios(candles, snap, zone_mult_atr, breakout_buffer_atr, vol_mult, use_ema200)
        return {
            "ok": True,
            "snapshot": asdict(snap),
            "scenarios": [asdict(s) for s in scs],
        }
    except Exception as e:
        engine.log("CONDITIONS_ERROR", error=str(e))
        return {"ok": False, "error": str(e)}

@app.get("/api/full_analysis")
async def api_full_analysis(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("3m"),
    limit: int = Query(500, ge=50, le=1000),
    capital: float = Query(10000, gt=0)
):
    """Endpoint para executar a análise completa dos 5 algoritmos."""
    try:
        # Reutilizar a lógica de snapshot do engine
        raw_klines = await fetch_klines(symbol, interval, limit)
        candles = parse_klines(raw_klines)

        # Executar a análise completa
        analysis_result = scalping_analyzer.analyze_all(candles, capital)

        return {"ok": True, **analysis_result}
    except Exception as e:
        return {"ok": False, "error": str(e)}