import time, asyncio, pandas as pd, webbrowser, threading, os, csv
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv
from pyquotex.stable_api import Quotex
from ia_engine import analyser_ia

load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
MIN_CONFIDENCE = 75
CSV_PATH = "data/candles.csv"

FOREX_ASSETS = ["EURUSD_otc","GBPUSD_otc","USDJPY_otc","USDCHF_otc","USDCAD_otc","AUDUSD_otc","NZDUSD_otc","EURJPY_otc","EURGBP_otc","GBPJPY_otc","AUDJPY_otc","EURAUD_otc","EURCAD_otc","GBPCAD_otc","GBPCHF_otc","CADJPY_otc","CHFJPY_otc","NZDJPY_otc"]

client = Quotex(email=EMAIL, password=PASSWORD)
signaux = []
live_data = {}
history_1m = defaultdict(lambda: deque(maxlen=2000))
selected_asset = FOREX_ASSETS[0]
status_msg = "Connexion..."
stable_counter = defaultdict(int)
last_action = defaultdict(str)
pending_until = {}

for a in FOREX_ASSETS:
    live_data[a] = {"rsi": "-", "ema9": "-", "ema21": "-", "signal": "INIT...", "count": 0, "raison": "-", "confiance": 0, "delai": 0, "expiration": "-", "sma20":"-","sma50":"-","macd":"-","macd_hist":"-","adx":"-","atr_pct":"-","m5_trend":"-","last_candle_time":"-","last_update":"-"}

def parse_exp(exp_str):
    try: return int(exp_str.replace("m","").strip()) * 60
    except: return 180

def build_5m(df_1m):
    df = df_1m.copy()
    df['bucket'] = (df['time'] // 300) * 300
    g = df.groupby('bucket')
    return pd.DataFrame({'time':g['time'].last(),'open':g['open'].first(),'high':g['high'].max(),'low':g['low'].min(),'close':g['close'].last()}).reset_index(drop=True)

def save_candle_csv(asset, c):
    try:
        os.makedirs("data", exist_ok=True)
        need_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH)==0
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if need_header: w.writerow(["time","asset","open","high","low","close"])
            w.writerow([int(c['time']), asset, c['open'], c['high'], c['low'], c['close']])
    except: pass

HTML_PAGE = """
<!DOCTYPE html>
<html><head><title>QUOTEX IA V10.12c</title><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0a0f1e;--card:#141e33;--card2:#1a2744;--border:#243656;--text:#e2e8f0;--muted:#94a3b8;--green:#22c55e;--red:#ef4444;--orange:#f59e0b;--blue:#3b82f6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:18px}
.topbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px 16px}
.brand h1{margin:0;font-size:18px;letter-spacing:0.4px}
.brand p{margin:2px 0 0;color:var(--muted);font-size:12px}
.pills{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.pill{padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid var(--border);background:#0f172a;color:var(--muted)}
.pill.live{color:#fff;background:var(--card2)}
.pill.ok{border-color:rgba(34,197,94,.4);color:var(--green)} .pill.bad{border-color:rgba(239,68,68,.4);color:var(--red)} .pill.warn{border-color:rgba(245,158,11,.4);color:var(--orange)}
select#assetSelect{padding:8px 12px;border-radius:10px;background:#0f172a;color:#fff;border:1px solid var(--border);font-weight:700}
.grid{display:grid;grid-template-columns:1.35fr .75fr;gap:14px;margin-top:14px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}
.card h3{margin:0 0 10px;font-size:13px;letter-spacing:.6px;color:var(--muted);text-transform:uppercase}
.signal-card{border-left:6px solid var(--border)}
.signal-card.CALL{border-left-color:var(--green)} .signal-card.PUT{border-left-color:var(--red)} .signal-card.WAIT{border-left-color:var(--orange)}
.big{font-size:34px;font-weight:900;letter-spacing:1px;margin:2px 0 6px}
.big.CALL{color:var(--green)} .big.PUT{color:var(--red)} .big.WAIT{color:var(--orange)}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.badge{padding:5px 10px;border-radius:999px;font-size:11px;font-weight:800}
.badge-exp{background:var(--blue);color:#fff} .badge-win{background:var(--green);color:#fff} .badge-lose{background:var(--red);color:#fff} .badge-pending{background:#334155;color:#cbd5e1} .badge-tie{background:#64748b;color:#fff}
.badge-neutral{background:#1e293b;color:var(--muted);border:1px solid var(--border)}
.reason{font-size:13px;line-height:1.4;color:#cbd5e1;background:#0f172a;border:1px solid var(--border);border-radius:10px;padding:10px}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.kv div{background:#0f172a;border:1px solid var(--border);border-radius:10px;padding:8px 10px}
.kv label{display:block;font-size:10px;color:var(--muted);letter-spacing:.5px;text-transform:uppercase}
.kv b{font-size:13px}
.indicators{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
@media(max-width:1100px){.indicators{grid-template-columns:repeat(3,1fr)}}
@media(max-width:600px){.indicators{grid-template-columns:repeat(2,1fr)}}
.ind{background:#0f172a;border:1px solid var(--border);border-radius:12px;padding:10px}
.ind label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.ind b{font-size:14px;display:block;margin-top:2px}
.ind small{color:var(--muted);font-size:11px}
.trend-up{color:var(--green)} .trend-down{color:var(--red)} .trend-flat{color:var(--muted)}
.market-row{display:flex;justify-content:space-between;font-size:13px;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.06)}
.market-row:last-child{border:0}
.progress{height:8px;background:#0f172a;border:1px solid var(--border);border-radius:999px;overflow:hidden;margin-top:8px}
.progress>div{height:100%;background:linear-gradient(90deg,var(--blue),#22c55e);width:0%}
.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:12px}
table{width:100%;border-collapse:collapse;min-width:760px;background:var(--card)}
th{font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:var(--muted);background:#0f172a;padding:10px 8px;text-align:left;position:sticky;top:0}
td{padding:10px 8px;border-top:1px solid rgba(255,255,255,.06);font-size:13px}
tr.win{border-left:3px solid var(--green)} tr.lose{border-left:3px solid var(--red)} tr.tie{border-left:3px solid #64748b} tr.pending{border-left:3px solid var(--border)}
.sig{font-weight:900} .sig.CALL{color:var(--green)} .sig.PUT{color:var(--red)} .sig.WAIT{color:var(--orange)}
details{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px}
details summary{cursor:pointer;color:var(--muted);font-weight:700}
pre{white-space:pre-wrap;word-break:break-word;background:#0f172a;border:1px solid var(--border);border-radius:10px;padding:10px;color:#cbd5e1;font-size:12px;max-height:240px;overflow:auto}
</style>
</head><body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <h1>QUOTEX IA — V10.12c <span style="color:var(--muted);font-weight:600">· M1 → 5m Forecast</span></h1>
      <p>Dashboard + Collecte CSV auto + Reload CSV</p>
    </div>
    <div class="pills">
      <span id="statusPill" class="pill warn">● Connexion...</span>
      <span class="pill live">Asset <b id="selLabel">-</b></span>
      <span class="pill">M1</span>
      <span class="pill">5m horizon</span>
      <span id="collectPill" class="pill">Collecte —</span>
      <select id="assetSelect" onchange="selectAsset(this.value)"></select>
    </div>
  </div>
  <div class="grid">
    <div class="card signal-card" id="signalCard"><h3>Signal Principal</h3><div id="signalMain">Chargement...</div></div>
    <div class="card" id="marketCard"><h3>Market Data</h3><div id="marketMain">Chargement...</div></div>
  </div>
  <div class="card" style="margin-top:14px"><h3>Indicateurs Compacts</h3><div class="indicators" id="indGrid"></div></div>
  <div class="card" style="margin-top:14px"><h3>Historique des Signaux <span id="statsLine" style="float:right;color:var(--muted);font-size:13px"></span></h3><div class="table-wrap"><table><thead><tr><th>Time</th><th>Asset</th><th>Signal</th><th>Evidence</th><th>Entry</th><th>Expiry</th><th>Result</th></tr></thead><tbody id="histBody"></tbody></table></div></div>
  <details style="margin-top:14px"><summary>Diagnostics</summary><div id="diag" style="margin-top:10px"></div></details>
</div>
<script>
async function selectAsset(v){ await fetch('/api/select_asset?asset='+encodeURIComponent(v)); }
function fmt(v,d=2){ if(v==="—"||v==="-"||v==null||v==="") return "—"; const n=Number(v); return isFinite(n)? n.toFixed(d) : String(v); }
function pillResult(r){ if(r==="WIN") return '<span class="badge badge-win">WIN</span>'; if(r==="LOSE") return '<span class="badge badge-lose">LOSS</span>'; if(r==="TIE") return '<span class="badge badge-tie">TIE</span>'; return '<span class="badge badge-pending">PENDING</span>'; }
async function load(){
  const res=await fetch('/api/signaux');const data=await res.json();
  const sel=document.getElementById('assetSelect');
  if(sel.options.length==0){data.assets.forEach(a=>{let o=document.createElement('option');o.value=a;o.textContent=a;sel.appendChild(o);});}
  sel.value=data.selected;document.getElementById('selLabel').textContent=data.selected;
  const st=(data.status||"").toLowerCase();
  const statusPill=document.getElementById('statusPill');
  statusPill.textContent=data.status+" · "+new Date().toLocaleTimeString();
  statusPill.className="pill "+(st.includes("echec")||st.includes("erreur")?"bad":st.includes("connexion")||st.includes("charg")?"warn":"ok");
  const d=data.live[data.selected]||{};
  const count=Number(d.count||0);const pct=Math.min(100,Math.round(count/80*100));
  document.getElementById('collectPill').textContent=count?count+" / 80 bougies":"Collecte —";
  const sig=(d.signal||"WAIT").toUpperCase();const conf=d.confiance??"—";const exp=d.expiration||"—";const raison=d.raison||"—";const cls=sig==="CALL"?"CALL":sig==="PUT"?"PUT":"WAIT";
  document.getElementById('signalCard').className="card signal-card "+cls;
  document.getElementById('signalMain').innerHTML=`<div class="big ${cls}">${sig}</div><div class="><span class="badge badge-neutral">Confiance <b>${fmt(conf,0)}${conf==="—"?"":"%"}</b></span><span class="badge badge-neutral">Bull <b>${fmt(d.bull_score,1)}</b> · Bear <b>${fmt(d.bear_score,1)}</b></span><span class="badge badge-exp">EXP ${exp}</span></div><div class="reason">${raison}</div><div class="kv"><div><label>Expiration</label><b>${exp}</b></div><div><label>Délai</label><b>${d.delai??"—"} min</b></div><div><label>Confiance</label><b>${fmt(conf,0)}${conf==="—"?"":"%"}</b></div><div><label>Status</label><b class="sig ${cls}">${sig}</b></div></div>`;
  document.getElementById('marketMain').innerHTML=`<div class="market-row"><span>M1 bougies</span><b>${count}</b></div><div class="market-row"><span>M5 bougies</span><b>${Math.floor(count/5)}</b></div><div class="market-row"><span>Dernière bougie</span><b>${d.last_candle_time||"—"}</b></div><div class="market-row"><span>Fraîcheur</span><b>${d.last_update||"—"}</b></div><div style="margin-top:10px"><div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted)"><span>Warm-up</span><span>${count} / 80</span></div><div class="progress"><div style="width:${pct}%"></div></div><div style="font-size:11px;color:var(--muted);margin-top:6px">${count<80?"Collecte en cours...":"Warm-up terminé ✓"}</div></div>`;
  const trend=(d.m5_trend||"—").toString();const trendCls=trend.includes("UP")?"trend-up":trend.includes("DOWN")?"trend-down":"trend-flat";
  document.getElementById('indGrid').innerHTML=[["RSI 14",fmt(d.rsi,1)],["EMA9",fmt(d.ema9,5)],["EMA21",fmt(d.ema21,5)],["SMA20",fmt(d.sma20,5)],["SMA50",fmt(d.sma50,5)],["MACD",fmt(d.macd,4)],["ADX",fmt(d.adx,1)],["ATR %",d.atr_pct!=="—"?fmt(d.atr_pct,3)+"%":"—"]].map(([k,v])=>`<div class="ind"><label>${k}</label><b>${v}</b></div>`).join("")+`<div class="ind"><label>M5 Trend</label><b class="${trendCls}">${trend}</b></div>`;
  const win=data.stats?.win??0,lose=data.stats?.lose??0,pending=data.stats?.pending??0;document.getElementById('statsLine').textContent=win+" WIN · "+lose+" LOSS · "+pending+" PENDING · WR "+(data.stats?.winrate??0)+"%";
  const body=document.getElementById('histBody');const rows=data.signaux||[];
  if(rows.length===0) body.innerHTML='<tr><td colspan="7" style="color:var(--muted);text-align:center;padding:18px">Aucun signal — analyse M1 → 5m</td></tr>';
  else body.innerHTML=rows.map(s=>{const r=(s.result||"PENDING").toUpperCase();const trCls=r==="WIN"?"win":r==="LOSE"?"lose":"pending";const sig2=(s.action||"—").toUpperCase();const ev=(s.confiance!=null?s.confiance+"% · ":"")+(s.raison||"—");return `<tr class="${trCls}"><td>${s.heure_signal||"—"}</td><td>${s.asset||"—"}</td><td class="sig ${sig2}">${sig2}</td><td style="max-width:280px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${ev}">${ev}</td><td>${s.heure_entree||"—"}</td><td>${s.expiration||"—"}</td><td>${pillResult(r)}</td></tr>`}).join("");
  document.getElementById('diag').innerHTML=`<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px"><div><label style="font-size:11px;color:var(--muted)">STATUS</label><pre>${data.status||"—"}</pre></div><div><label style="font-size:11px;color:var(--muted)">LIVE JSON</label><pre>${JSON.stringify(d,null,2)}</pre></div></div>`;
}
setInterval(load,1000);load();
</script></body></html>
"""

async def check_results_loop():
    while True:
        try:
            now=int(time.time())
            for s in signaux:
                if s['entry_price'] is None and now >= s['entry_ts'] + 2:
                    hist=list(history_1m.get(s['asset'], []))
                    if len(hist) > 0:
                        best=min(hist, key=lambda x: abs(x['time']-s['entry_ts']))
                        s['entry_price']=round(float(best['close']),5)
                if s['result'] != 'PENDING': continue
                if s['entry_price'] is None: continue
                exp_sec=parse_exp(s['expiration'])
                close_ts=s['entry_ts'] + exp_sec
                if now < close_ts + 5: continue
                hist=list(history_1m.get(s['asset'], []))
                if len(hist) < 5: continue
                best_close=min(hist, key=lambda x: abs(x['time']-close_ts))
                close_price=float(best_close['close'])
                entry_price=s['entry_price']
                s['close_price']=round(close_price,5)
                s['close_time']=time.strftime("%H:%M:%S", time.localtime(close_ts))
                if s['action']=='CALL': s['result']='WIN' if close_price > entry_price else 'LOSE'
                else: s['result']='WIN' if close_price < entry_price else 'LOSE'
        except Exception as e: print("check err",e)
        await asyncio.sleep(1)

async def boucle_quotex():
    global status_msg
    while True:
        try:
            check, reason = await client.connect()
            if not check: status_msg=f"Echec {reason}"; await asyncio.sleep(5); continue
            
            # 1. RELOAD CSV SI EXISTE - FIX 1/80
            if os.path.exists(CSV_PATH):
                try:
                    df_csv = pd.read_csv(CSV_PATH)
                    print(f"📂 Rechargement CSV: {len(df_csv)} lignes")
                    for asset in FOREX_ASSETS:
                        sub = df_csv[df_csv['asset']==asset].sort_values('time').tail(80)
                        for _, r in sub.iterrows():
                            c = {"time":int(r['time']), "open":float(r['open']), "high":float(r['high']), "low":float(r['low']), "close":float(r['close'])}
                            history_1m[asset].append(c)
                    print(f"✅ CSV rechargé: {len([a for a in history_1m if len(history_1m[a])>0])} assets restaurés")
                except Exception as e: print(f"CSV reload err {e}")

            # 2. COMPLETE AVEC QUOTEX
            status_msg = "Chargement historique Quotex..."
            for asset in FOREX_ASSETS:
                if len(history_1m[asset]) >= 80: continue
                try:
                    candles = await client.get_candles(asset, 60, 30, int(time.time()))
                    if candles:
                        for c in candles:
                            c['close'] = float(c['close']); c['open']=float(c['open']); c['high']=float(c['high']); c['low']=float(c['low'])
                            if 'time' in c: c['time'] = int(c['time']) // 60 * 60
                            elif 'from' in c: c['time'] = int(c['from']) // 60 * 60
                            else: c['time'] = int(time.time()) // 60 * 60
                            if not history_1m[asset] or history_1m[asset][-1]['time'] != c['time']:
                                history_1m[asset].append(c)
                    await asyncio.sleep(0.3)
                except Exception as e: print(f"histo err {asset} {e}")

            while True:
                now_ts=int(time.time()) // 60 * 60
                for asset in FOREX_ASSETS:
                    try:
                        candles=await client.get_candles(asset,60,1,now_ts)
                        if candles and len(candles)>0:
                            c=candles[0]
                            c['time'] = int(c.get('time') or c.get('from') or now_ts) // 60 * 60
                            c['close'] = float(c['close']); c['open']=float(c['open']); c['high']=float(c['high']); c['low']=float(c['low'])
                            hist = history_1m[asset]
                            is_new = not hist or hist[-1]['time'] != c['time']
                            if is_new:
                                hist.append(c)
                                save_candle_csv(asset, c)
                            else:
                                hist[-1]=c
                    except: pass
                    await asyncio.sleep(0.15)

                for asset in FOREX_ASSETS:
                    if asset in pending_until and int(time.time()) < pending_until[asset]: continue
                    cnt = len(history_1m[asset])
                    if cnt < 30:
                        live_data[asset]={"rsi":"-","ema9":"-","ema21":"-","signal":f"CHARGEMENT {cnt}/30","count":cnt,"raison":"Chargement...","confiance":0,"delai":0,"expiration":"-","sma20":"-","sma50":"-","macd":"-","macd_hist":"-","adx":"-","atr_pct":"-","m5_trend":"-","last_candle_time":"-","last_update":"-"}
                        continue
                    try:
                        df_1m=pd.DataFrame(list(history_1m[asset]))
                        for col in ['open','close','high','low']: df_1m[col]=pd.to_numeric(df_1m[col], errors='coerce')
                        df_5m=build_5m(df_1m) if len(df_1m)>=10 else df_1m.copy()
                        res=analyser_ia(df_1m, df_5m)
                        try:
                            last_row = df_1m.iloc[-1]
                            last_time = int(history_1m[asset][-1]['time']) if len(history_1m[asset])>0 else int(time.time())
                            rsi_v=round(float(last_row['RSI']),1) if 'RSI' in last_row and not pd.isna(last_row['RSI']) else "-"
                            ema9_v=round(float(last_row['EMA9']),5) if 'EMA9' in last_row and not pd.isna(last_row['EMA9']) else "-"
                            ema21_v=round(float(last_row['EMA21']),5) if 'EMA21' in last_row and not pd.isna(last_row['EMA21']) else "-"
                            sma20_v=round(float(last_row['SMA20']),5) if 'SMA20' in last_row and not pd.isna(last_row['SMA20']) else "-"
                            sma50_v=round(float(last_row['SMA50']),5) if 'SMA50' in last_row and not pd.isna(last_row['SMA50']) else "-"
                            macd_v=round(float(last_row['MACD']),5) if 'MACD' in last_row and not pd.isna(last_row['MACD']) else "-"
                            macd_hist_v=round(float(last_row['MACD_HIST']),5) if 'MACD_HIST' in last_row and not pd.isna(last_row['MACD_HIST']) else "-"
                            adx_v=round(float(last_row['ADX']),1) if 'ADX' in last_row and not pd.isna(last_row['ADX']) else "-"
                            atr_pct_v=round(float(last_row['ATR_PCT']),3) if 'ATR_PCT' in last_row and not pd.isna(last_row['ATR_PCT']) else "-"
                            m5_trend_v = "UP" if df_5m.iloc[-1]['EMA9'] > df_5m.iloc[-1]['EMA21'] else "DOWN" if df_5m.iloc[-1]['EMA9'] < df_5m.iloc[-1]['EMA21'] else "NEUTRE"
                            last_candle_str = time.strftime("%H:%M:%S", time.localtime(last_time))
                            fresh_str = f"{int(time.time())-last_time}s ago"
                            extra_dashboard = {"sma20":sma20_v,"sma50":sma50_v,"macd":macd_v,"macd_hist":macd_hist_v,"adx":adx_v,"atr_pct":atr_pct_v,"m5_trend":m5_trend_v,"last_candle_time":last_candle_str,"last_update":fresh_str,"time":last_time}
                            if res:
                                for k in ["rsi","ema9","ema21","sma20","sma50","macd","macd_hist","adx","atr_pct","m5_trend"]:
                                    if k in res and res[k] not in [None,"-"]: extra_dashboard[k]=res[k]
                                    if k=="rsi" and "rsi" in res: rsi_v=res["rsi"]
                        except: extra_dashboard={"sma20":"-","sma50":"-","macd":"-","macd_hist":"-","adx":"-","atr_pct":"-","m5_trend":"-","last_candle_time":"-","last_update":"-"}
                        if not res:
                            live_data[asset]={"rsi":rsi_v,"ema9":ema9_v,"ema21":ema21_v,"signal":"WAIT","count":cnt,"raison":"Pas de setup","confiance":0,"delai":0,"expiration":"-","bull_score":"-","bear_score":"-", **extra_dashboard}
                            stable_counter[asset]=0; last_action[asset]=""
                        elif res['confiance'] < MIN_CONFIDENCE:
                            live_data[asset]={"rsi":rsi_v,"ema9":ema9_v,"ema21":ema21_v,"signal":"WAIT","count":cnt,"raison":f"{res['action']} {res['confiance']}% < {MIN_CONFIDENCE}%","confiance":res['confiance'],"delai":res['delai'],"expiration":res['expiration'],"bull_score":res.get("bull_score","-"),"bear_score":res.get("bear_score","-"), **extra_dashboard}
                            stable_counter[asset]=0; last_action[asset]=""
                        else:
                            if res['action']==last_action[asset]: stable_counter[asset]+=1
                            else: stable_counter[asset]=1; last_action[asset]=res['action']
                            if stable_counter[asset] < 2:
                                live_data[asset]={"rsi":rsi_v,"ema9":ema9_v,"ema21":ema21_v,"signal":"WAIT","count":cnt,"raison":f"Confirmation 1/2 {res['action']} {res['confiance']}%","confiance":res['confiance'],"delai":res['delai'],"expiration":res['expiration'],"bull_score":res.get("bull_score","-"),"bear_score":res.get("bear_score","-"), **extra_dashboard}
                            else:
                                live_data[asset]={"rsi":rsi_v,"ema9":ema9_v,"ema21":ema21_v,"signal":res['action'],"count":cnt,"raison":res['raison'],"confiance":res['confiance'],"delai":res['delai'],"expiration":res['expiration'],"bull_score":res.get("bull_score","-"),"bear_score":res.get("bear_score","-"), **extra_dashboard}
                                delai=res['delai']; entry_ts=((int(time.time()) // 60) + delai + 1) * 60; entry_str=time.strftime("%H:%M:%S", time.localtime(entry_ts))
                                if not any(s['asset']==asset and abs(s['entry_ts']-entry_ts)<120 for s in signaux):
                                    sig={"asset":asset,"action":res['action'],"expiration":res['expiration'],"confiance":res['confiance'],"raison":res['raison'],"heure_signal":time.strftime("%H:%M:%S"),"heure_entree":entry_str,"entry_ts":entry_ts,"timestamp":time.time(),"delai":delai,"result":"PENDING","entry_price":None,"close_price":None,"close_time":None}
                                    signaux.insert(0,sig)
                                    pending_until[asset]=entry_ts + 300
                                    stable_counter[asset]=0
                    except Exception as e: print(e)
                csv_n = sum(1 for _ in open(CSV_PATH)) -1 if os.path.exists(CSV_PATH) else 0
                status_msg=f"✅ Multi 18 assets - {len([a for a in history_1m if len(history_1m[a])>=30])}/18 prêts | CSV {csv_n} bougies"
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Reconnexion {e}")
            try: await client.close()
            except: pass
            await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    t1=asyncio.create_task(boucle_quotex())
    t2=asyncio.create_task(check_results_loop())
    yield
    t1.cancel(); t2.cancel()

app=FastAPI(lifespan=lifespan)
@app.get("/", response_class=HTMLResponse)
async def dashboard(): return HTMLResponse(content=HTML_PAGE)
@app.get("/api/signaux")
async def get_signaux():
    win=sum(1 for s in signaux if s['result']=='WIN'); lose=sum(1 for s in signaux if s['result']=='LOSE'); pending=sum(1 for s in signaux if s['result']=='PENDING')
    winrate=round(win/(win+lose)*100,1) if (win+lose)>0 else 0
    return {"signaux":signaux,"live":live_data,"status":status_msg,"selected":selected_asset,"assets":FOREX_ASSETS,"stats":{"win":win,"lose":lose,"pending":pending,"winrate":winrate}}
@app.get("/api/select_asset")
async def select_asset(asset: str):
    global selected_asset
    if asset in FOREX_ASSETS: selected_asset=asset; return {"ok":True}
    return {"ok":False}
if __name__=="__main__":
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8001")).start()
    uvicorn.run(app, host="127.0.0.1", port=8001)