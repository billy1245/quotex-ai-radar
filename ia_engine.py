import pandas as pd
import numpy as np
import os
import joblib

# --- CHARGEMENT IA (si model/model.pkl existe) ---
MODEL = None
FEATURES = None
try:
    if os.path.exists("model/model.pkl") and os.path.exists("model/features.pkl"):
        MODEL = joblib.load("model/model.pkl")
        FEATURES = joblib.load("model/features.pkl")
        print(f"✅ IA chargée: {type(MODEL).__name__} avec {len(FEATURES)} features")
    else:
        print("ℹ️ Pas de model.pkl -> mode règles V11 PRO")
except Exception as e:
    print(f"IA non chargée: {e}")
    MODEL = None
    FEATURES = None

# --- INDICATEURS ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_val)
    dx = 100 * ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) )
    adx_val = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return plus_di, minus_di, adx_val

def analyser_ia(df_1m, df_5m):
    if len(df_1m) < 80:
        return None

    df1 = df_1m.copy()
    df5 = df_5m.copy() if len(df_5m) >= 14 else None
    use_5m = df5 is not None

    for df in [df1, df5] if use_5m else [df1]:
        df['RSI'] = rsi(df['close'], 14)
        df['EMA9'] = ema(df['close'], 9)
        df['EMA21'] = ema(df['close'], 21)
        df['SMA20'] = sma(df['close'], 20)
        df['SMA50'] = sma(df['close'], 50)
        df['ATR'] = atr(df['high'], df['low'], df['close'], 14)
        df['MACD'], df['MACD_SIG'], df['MACD_HIST'] = macd(df['close'])
        df['PLUS_DI'], df['MINUS_DI'], df['ADX'] = adx(df['high'], df['low'], df['close'])
        df['ATR_PCT'] = df['ATR'] / df['close'] * 100
        df['BB_MID'] = sma(df['close'], 20)
        df['BB_STD'] = df['close'].rolling(20).std()
        df['BB_UP'] = df['BB_MID'] + 2*df['BB_STD']
        df['BB_LOW'] = df['BB_MID'] - 2*df['BB_STD']

    last = df1.iloc[-1]
    prev = df1.iloc[-2]
    last5 = df5.iloc[-1] if use_5m else last

    if pd.isna(last['RSI']) or pd.isna(last['ADX']) or pd.isna(last['SMA50']):
        return None

    # --- SCORING PONDERE 9 FACTEURS ---
    bull = 0.0
    bear = 0.0
    reasons = []

    # 1. RSI
    if last['RSI'] < 30: bull += 1.5; reasons.append(f"RSI survente {last['RSI']:.1f}")
    elif last['RSI'] < 45: bull += 0.7
    elif last['RSI'] > 70: bear += 1.5; reasons.append(f"RSI surachat {last['RSI']:.1f}")
    elif last['RSI'] > 55: bear += 0.7
    else: reasons.append(f"RSI neutre {last['RSI']:.1f}")

    # 2. EMA9/21
    if last['EMA9'] > last['EMA21']: bull += 1.5; reasons.append("EMA9>EMA21 haussier")
    else: bear += 1.5; reasons.append("EMA9<EMA21 baissier")

    # 3. Prix vs EMA9
    if last['close'] > last['EMA9']: bull += 1.0
    else: bear += 1.0

    # 4. SMA20/50
    if last['SMA20'] > last['SMA50']: bull += 1.0; reasons.append("SMA20>SMA50 haussière")
    else: bear += 1.0; reasons.append("SMA20<SMA50 baissière")

    # 5. MACD
    if last['MACD'] > last['MACD_SIG'] and last['MACD_HIST'] > prev['MACD_HIST']: bull += 1.2; reasons.append("MACD haussier")
    elif last['MACD'] < last['MACD_SIG'] and last['MACD_HIST'] < prev['MACD_HIST']: bear += 1.2; reasons.append("MACD baissier")

    # 6. ADX filtre
    adx_val = last['ADX']
    if adx_val < 15:
        bull *= 0.7; bear *= 0.7
        reasons.append(f"ADX faible {adx_val:.1f} - marché mou")
    elif adx_val > 25:
        if last['PLUS_DI'] > last['MINUS_DI']: bull += 0.5
        else: bear += 0.5
        reasons.append(f"ADX fort {adx_val:.1f}")

    # 7. Bollinger
    if last['close'] < last['BB_LOW']: bull += 0.8; reasons.append("Sous Bollinger basse")
    elif last['close'] > last['BB_UP']: bear += 0.8; reasons.append("Sur Bollinger haute")

    # 8. M5 Trend confirmation - FILTRE MAJEUR
    m5_up = last5['EMA9'] > last5['EMA21'] and last5['close'] > last5['EMA9']
    m5_down = last5['EMA9'] < last5['EMA21'] and last5['close'] < last5['EMA9']
    m5_trend = "UP" if m5_up else "DOWN" if m5_down else "NEUTRE"
    if m5_up: bull += 1.5
    elif m5_down: bear += 1.5

    # --- DECISION ---
    total = bull + bear
    if total == 0:
        return None
    
    diff = abs(bull - bear)
    if diff < 1.2:
        return None
    
    if bull > bear and m5_down and adx_val > 20:
        return None
    if bear > bull and m5_up and adx_val > 20:
        return None

    if bull > bear:
        signal = "CALL"
        evidence = bull
    else:
        signal = "PUT"
        evidence = bear

    base = 58 + (diff * 7) + min(adx_val, 30) * 0.3
    if last['RSI'] < 25 or last['RSI'] > 75: base += 4
    confiance = int(min(92, max(55, base)))

    atr_pct = last['ATR_PCT']
    if last['RSI'] < 12 or last['RSI'] > 88: expiration = "15m"
    elif last['RSI'] < 20 or last['RSI'] > 80: expiration = "10m"
    elif last['RSI'] < 28 or last['RSI'] > 72: expiration = "5m"
    elif atr_pct < 0.03: expiration = "1m"
    elif atr_pct > 0.09: expiration = "2m"
    else: expiration = "3m"

    if confiance >= 86: delai = 1
    elif confiance >= 78: delai = 2
    else: delai = 3

    raison_str = f"Score BULL {bull:.1f} / BEAR {bear:.1f} | ADX {adx_val:.1f} | " + " · ".join(reasons[:3])

    # --- BOOST IA si modèle disponible ---
    extra_ia = {}
    if MODEL is not None and FEATURES is not None:
        try:
            bb_pos = (last['close']-last['BB_LOW'])/(last['BB_UP']-last['BB_LOW']) if last['BB_UP'] != last['BB_LOW'] else 0.5
            feat_map = {
                'RSI': last['RSI'],
                'EMA_DIFF': (last['EMA9']-last['EMA21'])/last['close']*100,
                'PRICE_EMA9': (last['close']-last['EMA9'])/last['close']*100,
                'SMA_DIFF': (last['SMA20']-last['SMA50'])/last['close']*100,
                'MACD': last['MACD'],
                'MACD_HIST': last['MACD_HIST'],
                'ADX': adx_val,
                'ATR_PCT': atr_pct,
                'BB_POS': bb_pos,
                'PLUS_DI': last['PLUS_DI'],
                'MINUS_DI': last['MINUS_DI']
            }
            vec = np.array([[feat_map[f] for f in FEATURES]])
            proba_call = float(MODEL.predict_proba(vec)[0][1])
            proba = proba_call if signal=="CALL" else 1-proba_call
            if proba < 0.52:
                return None
            confiance = int(min(92, confiance * 0.6 + proba*100 * 0.4 + 5))
            raison_str += f" | IA {proba*100:.0f}%"
            extra_ia = {"ia_proba": round(proba*100,1), "ia_model": type(MODEL).__name__}
        except Exception as e:
            extra_ia = {}

    return {
        "action": signal,
        "confiance": confiance,
        "expiration": expiration,
        "raison": raison_str,
        "delai": delai,
        "bull_score": round(bull,1),
        "bear_score": round(bear,1),
        "rsi": round(float(last['RSI']),1),
        "ema9": round(float(last['EMA9']),5),
        "ema21": round(float(last['EMA21']),5),
        "sma20": round(float(last['SMA20']),5),
        "sma50": round(float(last['SMA50']),5),
        "macd": round(float(last['MACD']),5),
        "macd_hist": round(float(last['MACD_HIST']),5),
        "adx": round(float(adx_val),1),
        "atr_pct": round(float(atr_pct),3),
        "m5_trend": m5_trend,
        **extra_ia
    }