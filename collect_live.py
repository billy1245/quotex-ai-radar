import asyncio, time, os, csv, pandas as pd
from dotenv import load_dotenv
from pyquotex.stable_api import Quotex

load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
ASSETS = ["EURUSD_otc","GBPUSD_otc","USDJPY_otc","EURJPY_otc","GBPJPY_otc","AUDUSD_otc"]
os.makedirs("data", exist_ok=True)
CSV = "data/candles.csv"

# Crée le header si vide
if not os.path.exists(CSV):
    with open(CSV,"w",newline="") as f:
        csv.writer(f).writerow(["time","asset","open","high","low","close"])

async def main():
    client = Quotex(email=EMAIL, password=PASSWORD)
    ok, msg = await client.connect()
    if not ok:
        print(f"Connexion échouée: {msg}")
        return
    print(f"✅ Collecteur connecté: {msg}")
    print(f"📝 Enregistrement dans {CSV} - laisse tourner 24h")
    seen = set()
    # Charge déjà vu pour éviter doublons au restart
    if os.path.exists(CSV):
        try:
            df = pd.read_csv(CSV)
            seen = set(zip(df['time'], df['asset']))
            print(f"Reprise: {len(seen)} bougies déjà collectées")
        except: pass

    while True:
        now = int(time.time()) // 60 * 60  # aligne à la minute
        for asset in ASSETS:
            try:
                c = await client.get_candles(asset, 60, 1, now)
                if c and len(c)>0:
                    candle = c[0]
                    t = int(candle.get('time') or candle.get('from') or now)
                    t = t // 60 * 60
                    key = (t, asset)
                    if key not in seen:
                        row = [t, asset, candle['open'], candle['high'], candle['low'], candle['close']]
                        with open(CSV,"a",newline="") as f:
                            csv.writer(f).writerow(row)
                        seen.add(key)
                        print(f"+ {asset} {time.strftime('%H:%M:%S', time.localtime(t))} {candle['close']} | total {len(seen)}")
                    else:
                        print(f"= {asset} {time.strftime('%H:%M:%S', time.localtime(t))} déjà enregistré")
                else:
                    print(f"- {asset} vide à {time.strftime('%H:%M:%S', time.localtime(now))}")
            except Exception as e:
                print(f"err {asset} {e}")
            await asyncio.sleep(0.4)
        # Attend la prochaine minute pile + 5s
        wait = 65 - (int(time.time()) % 60)
        print(f"--- {len(seen)} bougies totales | prochain scan dans {wait}s ---\n")
        await asyncio.sleep(wait)

if __name__ == "__main__":
    asyncio.run(main())