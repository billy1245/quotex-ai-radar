import os, pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except:
    from sklearn.ensemble import RandomForestClassifier
    USE_XGB = False

from ia_engine import rsi, ema, sma, atr, macd, adx

CSV = "data/candles.csv"
FORECAST = 5

def build_features(df):
    df = df.copy()
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
    df['EMA_DIFF'] = (df['EMA9'] - df['EMA21']) / df['close'] * 100
    df['PRICE_EMA9'] = (df['close'] - df['EMA9']) / df['close'] * 100
    df['SMA_DIFF'] = (df['SMA20'] - df['SMA50']) / df['close'] * 100
    df['BB_POS'] = (df['close'] - df['BB_LOW']) / (df['BB_UP'] - df['BB_LOW']).replace(0, np.nan)
    return df

def train():
    if not os.path.exists(CSV):
        print(f"Pas de {CSV}")
        print("Lance d'abord: python3 collect_live.py et laisse tourner 24h")
        return
    df_all = pd.read_csv(CSV)
    print(f"CSV brut: {len(df_all)} lignes")
    if len(df_all) < 500:
        print(f"Trop peu de données {len(df_all)}/500. Laisse tourner collect_live.py encore quelques heures.")
        print(f"Actuel: {len(df_all)} -> besoin 500 mini, idéal 2000")
        return

    all_dfs = []
    for asset in df_all['asset'].unique():
        df = df_all[df_all['asset']==asset].sort_values('time').copy()
        for c in ['open','high','low','close']: df[c] = pd.to_numeric(df[c], errors='coerce')
        df = build_features(df)
        df['future_close'] = df['close'].shift(-FORECAST)
        df['label'] = (df['future_close'] > df['close']).astype(int)
        df = df.dropna(subset=['RSI','SMA50','ADX','BB_POS','future_close'])
        if len(df) > 80:
            all_dfs.append(df)
            print(f"  {asset}: {len(df)} lignes OK")

    if not all_dfs:
        print("Pas assez de données après warm-up 80 bougies")
        return

    full = pd.concat(all_dfs, ignore_index=True)
    features = ['RSI','EMA_DIFF','PRICE_EMA9','SMA_DIFF','MACD','MACD_HIST','ADX','ATR_PCT','BB_POS','PLUS_DI','MINUS_DI']
    X = full[features].values
    y = full['label'].values
    print(f"\nDataset total: {len(full)} lignes | CALL {sum(y)} / PUT {len(y)-sum(y)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    if USE_XGB:
        print("Entraînement XGBClassifier...")
        model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', n_jobs=-1)
    else:
        print("Fallback RandomForest...")
        model = RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, pred)
    print(f"\nAccuracy test: {acc*100:.2f}%")
    print(classification_report(y_test, pred, target_names=['PUT','CALL']))
    mask = (proba > 0.60) | (proba < 0.40)
    if sum(mask) > 20:
        acc_filtre = accuracy_score(y_test[mask], pred[mask])
        print(f"Accuracy filtre >60%: {acc_filtre*100:.2f}% sur {sum(mask)} trades filtrés")
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(features, "model/features.pkl")
    print("\n✅ Modèle sauvé dans model/model.pkl")
    try:
        imp = model.feature_importances_
        for f,i in sorted(zip(features, imp), key=lambda x: -x[1]):
            print(f"  {f:12s} {i*100:.1f}%")
    except: pass
    print("\nRelance: python3 main.py -> ✅ IA chargée")

if __name__ == "__main__":
    train()