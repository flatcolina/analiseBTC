from typing import List, Optional
from models import Candle

def ema_series(values: list[float], period: int) -> list[Optional[float]]:
    if period <= 1:
        return [float(v) for v in values]
    k = 2.0 / (period + 1.0)
    out: list[Optional[float]] = [None] * len(values)
    if not values:
        return out
    ema_val = values[0]
    out[0] = ema_val
    for i in range(1, len(values)):
        ema_val = values[i] * k + ema_val * (1.0 - k)
        out[i] = ema_val
    return out

def rsi_series(candles: list[Candle], period: int = 14) -> list[Optional[float]]:
    if len(candles) < period + 1:
        return [None] * len(candles)
    gains: list[float] = []
    losses: list[float] = []
    out: list[Optional[float]] = [None] * len(candles)
    # first average
    for i in range(1, period + 1):
        ch = candles[i].close - candles[i - 1].close
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
    out[period] = 100.0 - (100.0 / (1.0 + rs))
    # Wilder smoothing
    for i in range(period + 1, len(candles)):
        ch = candles[i].close - candles[i - 1].close
        g = max(ch, 0.0)
        l = max(-ch, 0.0)
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        rs = (avg_gain / avg_loss) if avg_loss > 0 else float("inf")
        out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out

def atr_wilder_series(candles: list[Candle], period: int = 14) -> list[Optional[float]]:
    if len(candles) < period + 1:
        return [None] * len(candles)
    trs: list[float] = []
    out: list[Optional[float]] = [None] * len(candles)
    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        pc = candles[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    # initial ATR = SMA of first period TRs
    atr = sum(trs[:period]) / period
    out[period] = atr
    for i in range(period + 1, len(candles)):
        tr = trs[i - 1]
        atr = (atr * (period - 1) + tr) / period
        out[i] = atr
    return out

def avg_volume(candles: list[Candle], period: int = 20) -> Optional[float]:
    if len(candles) < period:
        return None
    return sum(c.volume for c in candles[-period:]) / float(period)

def vwap_rolling(candles: list[Candle], period: int = 100) -> Optional[float]:
    if len(candles) < 2:
        return None
    use = candles[-period:] if len(candles) >= period else candles[:]
    pv = 0.0
    v = 0.0
    for c in use:
        typical = (c.high + c.low + c.close) / 3.0
        pv += typical * c.volume
        v += c.volume
    return (pv / v) if v > 0 else None

def macd_series(closes: list[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    ema_fast = ema_series(closes, fast_period)
    ema_slow = ema_series(closes, slow_period)
    
    macd_line = [ (f - s) if f is not None and s is not None else None for f, s in zip(ema_fast, ema_slow) ]
    
    # Remove None values for signal line calculation
    valid_macd_line = [m for m in macd_line if m is not None]
    signal_line_raw = ema_series(valid_macd_line, signal_period)
    
    # Re-align signal line with original macd_line
    signal_line = [None] * len(macd_line)
    signal_idx = 0
    for i, m in enumerate(macd_line):
        if m is not None:
            if signal_idx < len(signal_line_raw):
                signal_line[i] = signal_line_raw[signal_idx]
                signal_idx += 1

    histogram = [ (m - s) if m is not None and s is not None else None for m, s in zip(macd_line, signal_line) ]
    
    return macd_line, signal_line, histogram
