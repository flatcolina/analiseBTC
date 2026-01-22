
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

BINANCE_SPOT_BASE = "https://api.binance.com"

def now_ms() -> int:
    return int(time.time() * 1000)

def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")

def apply_slippage(price: float, side: str, slippage_bps: float) -> float:
    """Adverse slippage in bps. BUY => worse (higher); SELL => worse (lower)."""
    if slippage_bps <= 0:
        return price
    m = slippage_bps / 10000.0
    s = side.upper()
    if s == "BUY":
        return price * (1.0 + m)
    if s == "SELL":
        return price * (1.0 - m)
    return price

@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int

async def fetch_klines(symbol: str, interval: str, limit: int = 500) -> list[Candle]:
    url = f"{BINANCE_SPOT_BASE}/api/v3/klines"
    params: dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    out: list[Candle] = []
    for row in data:
        out.append(
            Candle(
                open_time=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time=int(row[6]),
            )
        )
    return out

async def fetch_price(symbol: str) -> float:
    url = f"{BINANCE_SPOT_BASE}/api/v3/ticker/price"
    params = {"symbol": symbol.upper()}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return float(r.json()["price"])

def atr_wilder(candles: list[Candle], period: int = 14) -> list[float | None]:
    n = len(candles)
    out: list[float | None] = [None] * n
    if n < period + 1:
        return out
    trs: list[float] = []
    for i in range(1, n):
        c = candles[i]
        p = candles[i - 1]
        tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
        trs.append(tr)
    seed = sum(trs[:period]) / period
    out[period] = seed
    prev = seed
    for j in range(period + 1, n):
        tr = trs[j - 1]
        prev = (prev * (period - 1) + tr) / period
        out[j] = prev
    return out

def rsi(candles: list[Candle], period: int = 14) -> list[float | None]:
    n = len(candles)
    out: list[float | None] = [None] * n
    if n < period + 1:
        return out
    gains = []
    losses = []
    for i in range(1, n):
        change = candles[i].close - candles[i - 1].close
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
    out[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period + 1, n):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
        out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out

def ema(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period or period <= 1:
        return out
    k = 2.0 / (period + 1.0)
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    prev = sma
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out

def vwap_rolling(candles: list[Candle], window: int = 100) -> float | None:
    if len(candles) < 5:
        return None
    w = candles[-window:] if len(candles) >= window else candles
    pv = 0.0
    vv = 0.0
    for c in w:
        typical = (c.high + c.low + c.close) / 3.0
        pv += typical * c.volume
        vv += c.volume
    if vv == 0:
        return None
    return pv / vv

def avg_volume(candles: list[Candle], window: int = 20) -> float | None:
    if len(candles) < window:
        return None
    vs = [c.volume for c in candles[-window:]]
    return sum(vs) / len(vs)

Direction = Literal["LONG", "SHORT"]
ScenarioType = Literal["PULLBACK", "BREAKOUT"]

@dataclass
class ScenarioDef:
    scenario_type: ScenarioType
    direction: Direction
    name: str
    if_then: str

@dataclass
class AnalysisSnapshot:
    ts_ms: int
    symbol: str
    interval: str
    price: float
    vwap: float | None
    ema200: float | None
    rsi14: float | None
    atr14: float | None
    avg_vol20: float | None
    recent_high: float | None
    recent_low: float | None

@dataclass
class EngineStats:
    cycles_total: int = 0
    no_entry: int = 0
    tp: int = 0
    sl: int = 0
    time: int = 0

class ScenarioEngine:
    def __init__(self) -> None:
        self.running = False
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.logs: list[dict[str, Any]] = []
        self.stats = EngineStats()
        self.last_snapshot: AnalysisSnapshot | None = None
        self.last_scenarios: list[ScenarioDef] = []
        self.last_cycle: dict[str, Any] | None = None
        self.cycle_id = 0

    def log(self, event: str, **data: Any) -> None:
        row = {"ts_ms": now_ms(), "ts": ms_to_iso(now_ms()), "event": event, **data}
        self.logs.append(row)
        if len(self.logs) > 2000:
            self.logs = self.logs[-2000:]

    async def start(self, **cfg: Any) -> None:
        async with self.lock:
            if self.running:
                return
            self.running = True
            self.task = asyncio.create_task(self._run(**cfg))
            self.log("ENGINE_START", cfg=cfg)

    async def stop(self) -> None:
        async with self.lock:
            self.running = False
            if self.task:
                self.task.cancel()
                self.task = None
            self.log("ENGINE_STOP")

    async def reset(self) -> None:
        async with self.lock:
            self.logs = []
            self.stats = EngineStats()
            self.last_snapshot = None
            self.last_scenarios = []
            self.last_cycle = None
            self.cycle_id = 0
            self.log("RESET_DONE")

    async def _snapshot(self, symbol: str, interval: str, limit: int) -> tuple[list[Candle], AnalysisSnapshot]:
        candles = await fetch_klines(symbol, interval=interval, limit=limit)
        price = candles[-1].close
        closes = [c.close for c in candles]
        ema200s = ema(closes, 200)
        rsis = rsi(candles, 14)
        atrs = atr_wilder(candles, 14)
        vwap = vwap_rolling(candles, 100)
        avgv = avg_volume(candles, 20)

        lookback = 20
        recent = candles[-lookback:] if len(candles) >= lookback else candles
        rh = max(c.high for c in recent) if recent else None
        rl = min(c.low for c in recent) if recent else None

        snap = AnalysisSnapshot(
            ts_ms=now_ms(),
            symbol=symbol.upper(),
            interval=interval,
            price=float(price),
            vwap=vwap,
            ema200=ema200s[-1],
            rsi14=rsis[-1],
            atr14=atrs[-1],
            avg_vol20=avgv,
            recent_high=rh,
            recent_low=rl,
        )
        return candles, snap

    def _build_scenarios(self, snap: AnalysisSnapshot, zone_mult_atr: float, breakout_buffer_atr: float, vol_mult: float, use_ema200: bool) -> list[ScenarioDef]:
        vwap = snap.vwap
        atr = snap.atr14
        rsi14 = snap.rsi14
        ema200 = snap.ema200
        rh = snap.recent_high
        rl = snap.recent_low

        if vwap is None or atr is None or rsi14 is None or rh is None or rl is None:
            return []

        zone = max(atr * zone_mult_atr, atr * 0.25)
        buf = atr * breakout_buffer_atr

        ema_long = f" E preço > EMA200 (~{ema200:.0f})" if (use_ema200 and ema200 is not None) else ""
        ema_short = f" E preço < EMA200 (~{ema200:.0f})" if (use_ema200 and ema200 is not None) else ""

        return [
            ScenarioDef(
                scenario_type="PULLBACK",
                direction="LONG",
                name="Cenário 1 — Pullback LONG (VWAP)",
                if_then=f"SE preço tocar VWAP±{zone:.0f} E voltar/fechar acima do VWAP E RSI(14) >= 50{ema_long} ENTÃO COMPRAR (LONG).",
            ),
            ScenarioDef(
                scenario_type="PULLBACK",
                direction="SHORT",
                name="Cenário 1 — Pullback SHORT (VWAP)",
                if_then=f"SE preço tocar VWAP±{zone:.0f} E voltar/fechar abaixo do VWAP E RSI(14) <= 50{ema_short} ENTÃO VENDER/SHORT (SHORT).",
            ),
            ScenarioDef(
                scenario_type="BREAKOUT",
                direction="LONG",
                name="Cenário 2 — Breakout LONG (confirmado)",
                if_then=f"SE candle fechar acima do topo recente ({rh:.0f}) + buffer({buf:.0f}) E acima do VWAP E RSI(14) >= 55 E volume >= {vol_mult:.2f}x média(20){ema_long} ENTÃO COMPRAR (LONG).",
            ),
            ScenarioDef(
                scenario_type="BREAKOUT",
                direction="SHORT",
                name="Cenário 2 — Breakout SHORT (confirmado)",
                if_then=f"SE candle fechar abaixo do fundo recente ({rl:.0f}) - buffer({buf:.0f}) E abaixo do VWAP E RSI(14) <= 45 E volume >= {vol_mult:.2f}x média(20){ema_short} ENTÃO VENDER/SHORT (SHORT).",
            ),
        ]

    def _check_trigger(
        self,
        candles: list[Candle],
        snap: AnalysisSnapshot,
        zone_mult_atr: float,
        breakout_buffer_atr: float,
        vol_mult: float,
        use_ema200: bool,
    ) -> ScenarioDef | None:
        if len(candles) < 25:
            return None
        last = candles[-1]
        prev = candles[-2]
        vwap = snap.vwap
        atr = snap.atr14
        rsi14 = snap.rsi14
        ema200 = snap.ema200
        avgv = snap.avg_vol20
        rh = snap.recent_high
        rl = snap.recent_low
        if vwap is None or atr is None or rsi14 is None or avgv is None or rh is None or rl is None:
            return None

        zone = max(atr * zone_mult_atr, atr * 0.25)
        buf = atr * breakout_buffer_atr

        def ema_ok(direction: Direction) -> bool:
            if not use_ema200 or ema200 is None:
                return True
            return (last.close > ema200) if direction == "LONG" else (last.close < ema200)

        # Pullback LONG
        touched = (prev.low <= vwap + zone) and (prev.high >= vwap - zone)
        if touched and last.close > vwap and rsi14 >= 50 and ema_ok("LONG"):
            return ScenarioDef("PULLBACK", "LONG", "Cenário 1 — Pullback LONG (VWAP)", "")

        # Pullback SHORT
        touched2 = (prev.high >= vwap - zone) and (prev.low <= vwap + zone)
        if touched2 and last.close < vwap and rsi14 <= 50 and ema_ok("SHORT"):
            return ScenarioDef("PULLBACK", "SHORT", "Cenário 1 — Pullback SHORT (VWAP)", "")

        # Breakout LONG
        if last.close > (rh + buf) and last.close > vwap and rsi14 >= 55 and last.volume >= (avgv * vol_mult) and ema_ok("LONG"):
            return ScenarioDef("BREAKOUT", "LONG", "Cenário 2 — Breakout LONG (confirmado)", "")

        # Breakout SHORT
        if last.close < (rl - buf) and last.close < vwap and rsi14 <= 45 and last.volume >= (avgv * vol_mult) and ema_ok("SHORT"):
            return ScenarioDef("BREAKOUT", "SHORT", "Cenário 2 — Breakout SHORT (confirmado)", "")

        return None

    async def _monitor_trade(
        self,
        symbol: str,
        direction: Direction,
        entry_price_raw: float,
        entry_exec: float,
        qty_btc: float,
        tp_atr_mult: float,
        sl_atr_mult: float,
        atr_value: float,
        slippage_bps: float,
        fee_rate_per_side: float,
        max_hold_minutes: int,
        poll_seconds: float,
    ) -> dict[str, Any]:
        tp_dist = atr_value * tp_atr_mult
        sl_dist = atr_value * sl_atr_mult
        tp_price = entry_price_raw + tp_dist if direction == "LONG" else entry_price_raw - tp_dist
        sl_price = entry_price_raw - sl_dist if direction == "LONG" else entry_price_raw + sl_dist

        self.log("TRADE_SET", direction=direction, entry_price=entry_price_raw, tp_price=tp_price, sl_price=sl_price, tp_dist=tp_dist, sl_dist=sl_dist)

        deadline = now_ms() + max_hold_minutes * 60_000

        while self.running:
            p = await fetch_price(symbol)
            t_ms = now_ms()

            hit_tp = (p >= tp_price) if direction == "LONG" else (p <= tp_price)
            hit_sl = (p <= sl_price) if direction == "LONG" else (p >= sl_price)

            if hit_tp or hit_sl or t_ms >= deadline:
                outcome = "TIME"
                exit_raw = p
                if hit_tp:
                    outcome = "TP"
                    exit_raw = tp_price
                elif hit_sl:
                    outcome = "SL"
                    exit_raw = sl_price

                exit_side = "SELL" if direction == "LONG" else "BUY"
                exit_exec = apply_slippage(exit_raw, exit_side, slippage_bps)

                gross = qty_btc * (exit_exec - entry_exec) if direction == "LONG" else qty_btc * (entry_exec - exit_exec)
                notional_entry = qty_btc * entry_exec
                notional_exit = qty_btc * exit_exec
                fees = (notional_entry + notional_exit) * fee_rate_per_side
                net = gross - fees

                return {
                    "outcome": outcome,
                    "exit_time_ms": t_ms,
                    "exit_time": ms_to_iso(t_ms),
                    "exit_price_raw": exit_raw,
                    "exit_price_exec": exit_exec,
                    "gross_pnl_usd": gross,
                    "fees_usd": fees,
                    "net_pnl_usd": net,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                }

            await asyncio.sleep(poll_seconds)

        # stopped
        t_ms = now_ms()
        return {
            "outcome": "TIME",
            "exit_time_ms": t_ms,
            "exit_time": ms_to_iso(t_ms),
            "exit_price_raw": entry_price_raw,
            "exit_price_exec": entry_exec,
            "gross_pnl_usd": 0.0,
            "fees_usd": 0.0,
            "net_pnl_usd": 0.0,
        }

    async def _run(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "3m",
        limit: int = 500,
        poll_seconds: float = 5.0,
        entry_timeout_minutes: int = 10,
        max_hold_minutes: int = 25,
        fee_rate_per_side: float = 0.001,   # 0.10% per order
        slippage_bps: float = 2.0,
        collateral_usd: float = 1879.0,
        zone_mult_atr: float = 0.35,
        breakout_buffer_atr: float = 0.20,
        vol_mult: float = 1.10,
        use_ema200: bool = True,
        tp_atr_mult: float = 1.0,
        sl_atr_mult: float = 0.7,
    ) -> None:
        self.log("ENGINE_LOOP_PARAMS", symbol=symbol, interval=interval, poll_seconds=poll_seconds,
                 entry_timeout_minutes=entry_timeout_minutes, max_hold_minutes=max_hold_minutes,
                 fee_rate_per_side=fee_rate_per_side, slippage_bps=slippage_bps, collateral_usd=collateral_usd)

        while self.running:
            self.cycle_id += 1
            cid = self.cycle_id
            cycle_start = now_ms()
            self.stats.cycles_total += 1

            try:
                candles, snap = await self._snapshot(symbol, interval, limit)
            except Exception as e:
                self.log("SNAPSHOT_ERROR", cycle_id=cid, error=str(e))
                await asyncio.sleep(2.0)
                continue

            self.last_snapshot = snap
            scenarios = self._build_scenarios(snap, zone_mult_atr, breakout_buffer_atr, vol_mult, use_ema200)
            self.last_scenarios = scenarios

            self.log("CYCLE_START", cycle_id=cid, snapshot=asdict(snap))
            for sc in scenarios:
                self.log("SCENARIO_DEF", cycle_id=cid, scenario=asdict(sc))

            deadline = cycle_start + entry_timeout_minutes * 60_000
            triggered: ScenarioDef | None = None

            while self.running and now_ms() < deadline and triggered is None:
                try:
                    candles, snap2 = await self._snapshot(symbol, interval, limit)
                    self.last_snapshot = snap2
                except Exception as e:
                    self.log("POLL_ERROR", cycle_id=cid, error=str(e))
                    await asyncio.sleep(poll_seconds)
                    continue

                triggered = self._check_trigger(candles, snap2, zone_mult_atr, breakout_buffer_atr, vol_mult, use_ema200)
                if triggered:
                    self.log("TRIGGERED", cycle_id=cid, triggered=asdict(triggered), price=snap2.price)
                    break
                await asyncio.sleep(poll_seconds)

            if not self.running:
                break

            if triggered is None:
                self.stats.no_entry += 1
                self.last_cycle = {"cycle_id": cid, "status": "NO_ENTRY", "duration_seconds": (now_ms()-cycle_start)/1000.0}
                self.log("CYCLE_END_NO_ENTRY", cycle_id=cid, duration_s=self.last_cycle["duration_seconds"])
                continue

            direction: Direction = triggered.direction
            entry_time = now_ms()
            raw_price = await fetch_price(symbol)
            entry_side = "BUY" if direction == "LONG" else "SELL"
            entry_exec = apply_slippage(raw_price, entry_side, slippage_bps)

            atr_val = (self.last_snapshot.atr14 if (self.last_snapshot and self.last_snapshot.atr14) else None)
            if atr_val is None:
                atrs = atr_wilder(candles, 14)
                atr_val = atrs[-1] or 0.0

            qty = collateral_usd / entry_exec if entry_exec > 0 else 0.0

            entry_info = {
                "entry_time_ms": entry_time,
                "entry_time": ms_to_iso(entry_time),
                "entry_price_raw": raw_price,
                "entry_price_exec": entry_exec,
                "direction": direction,
                "qty_btc": qty,
                "collateral_usd": collateral_usd,
                "fee_rate_per_side": fee_rate_per_side,
                "slippage_bps": slippage_bps,
                "atr14": float(atr_val),
            }
            self.log("ENTER", cycle_id=cid, **entry_info)

            exit_info = await self._monitor_trade(
                symbol=symbol,
                direction=direction,
                entry_price_raw=raw_price,
                entry_exec=entry_exec,
                qty_btc=qty,
                tp_atr_mult=tp_atr_mult,
                sl_atr_mult=sl_atr_mult,
                atr_value=float(atr_val),
                slippage_bps=slippage_bps,
                fee_rate_per_side=fee_rate_per_side,
                max_hold_minutes=max_hold_minutes,
                poll_seconds=poll_seconds,
            )

            self.log("EXIT", cycle_id=cid, **exit_info)

            status = "ENTERED_TIME"
            if exit_info["outcome"] == "TP":
                status = "ENTERED_TP"
                self.stats.tp += 1
            elif exit_info["outcome"] == "SL":
                status = "ENTERED_SL"
                self.stats.sl += 1
            else:
                self.stats.time += 1

            duration_s = (now_ms() - cycle_start) / 1000.0
            self.last_cycle = {
                "cycle_id": cid,
                "status": status,
                "duration_seconds": duration_s,
                "scenario_triggered": asdict(triggered),
                "entry": entry_info,
                "exit": exit_info,
            }
            self.log("CYCLE_END", cycle_id=cid, status=status, duration_s=duration_s)

engine = ScenarioEngine()

app = FastAPI(title="BTC Scenario Logger", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "ts": ms_to_iso(now_ms())}

@app.post("/api/start")
async def api_start(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("3m"),
    collateral_usd: float = Query(1879.0, gt=0),
    poll_seconds: float = Query(5.0, ge=1.0, le=30.0),
    entry_timeout_minutes: int = Query(10, ge=1, le=60),
    max_hold_minutes: int = Query(25, ge=1, le=240),
    fee_rate_per_side: float = Query(0.001, ge=0.0, le=0.01),
    slippage_bps: float = Query(2.0, ge=0.0, le=50.0),
    zone_mult_atr: float = Query(0.35, gt=0.0, le=2.0),
    breakout_buffer_atr: float = Query(0.20, gt=0.0, le=2.0),
    vol_mult: float = Query(1.10, gt=0.5, le=5.0),
    use_ema200: bool = Query(True),
    tp_atr_mult: float = Query(1.0, gt=0.1, le=10.0),
    sl_atr_mult: float = Query(0.7, gt=0.1, le=10.0),
):
    await engine.start(
        symbol=symbol,
        interval=interval,
        collateral_usd=collateral_usd,
        poll_seconds=poll_seconds,
        entry_timeout_minutes=entry_timeout_minutes,
        max_hold_minutes=max_hold_minutes,
        fee_rate_per_side=fee_rate_per_side,
        slippage_bps=slippage_bps,
        zone_mult_atr=zone_mult_atr,
        breakout_buffer_atr=breakout_buffer_atr,
        vol_mult=vol_mult,
        use_ema200=use_ema200,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
    )
    return {"running": engine.running}

@app.post("/api/stop")
async def api_stop():
    await engine.stop()
    return {"running": engine.running}

@app.post("/api/reset")
async def api_reset():
    await engine.reset()
    return {"ok": True}

@app.get("/api/status")
def api_status():
    return {
        "running": engine.running,
        "stats": asdict(engine.stats),
        "last_snapshot": asdict(engine.last_snapshot) if engine.last_snapshot else None,
        "last_scenarios": [asdict(s) for s in engine.last_scenarios],
        "last_cycle": engine.last_cycle,
        "logs_len": len(engine.logs),
    }

@app.get("/api/logs")
def api_logs(limit: int = Query(300, ge=1, le=2000)):
    return {"logs": engine.logs[-int(limit):]}
