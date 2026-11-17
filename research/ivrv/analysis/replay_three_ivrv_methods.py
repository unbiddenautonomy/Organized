"""Replay each live IV/RV route with an independent three-position book."""
import json
import os
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis_outputs'/os.environ.get('IVRV_OUTPUT_DIR','previous_day_ivrv_2026-10-29')
p=pd.read_csv(OUT/'predictions.csv',parse_dates=['datetime'])
o=pd.read_csv(ROOT/'Data'/'banknifty_atm5_minute.csv')
o.minute=pd.to_datetime(o.minute)
quotes={(r.minute,float(r.strike),r.option_type):(float(r.BestBuyPrice_1),float(r.BestSellPrice_1)) for r in o.itertuples()}
methods=['iv_above_rv_convergence','iv_near_rv_sharp_rv_decline','iv_below_rv_sharp_rv_decline']
signals={method:{r.datetime+pd.Timedelta(minutes=1):r for r in p.itertuples()
                 if r.paper_route==method and r.paper_signal and r.quality_entry_ok} for method in methods}
books={method:[] for method in methods}
trades=[]; skipped=[]
for minute in sorted(o.minute.unique()):
    minute=pd.Timestamp(minute)
    for method in methods:
        active=[]
        for pos in books[method]:
            call=quotes.get((minute,pos['strike'],'CE'))
            put=quotes.get((minute,pos['strike'],'PE'))
            if call is None or put is None:
                active.append(pos);continue
            pnl=(pos['entry_premium']-call[1]-put[1])*30
            held=int((minute-pos['entry_time']).total_seconds()/60)
            reason=('target' if pnl>=1500 else 'stop' if pnl<=-750 else
                    'horizon' if held>=120 else 'square_off' if minute.time()>=pd.Timestamp('15:14:30').time() else None)
            if reason:
                trades.append({**pos,'exit_time':minute,'exit_call_ask':call[1],
                               'exit_put_ask':put[1],'gross_pnl':pnl,'exit_reason':reason})
            else:active.append(pos)
        books[method]=active
        signal=signals[method].get(minute)
        if signal is None:continue
        if minute.time()>=pd.Timestamp('15:14:30').time():
            skipped.append({'method':method,'signal_time':signal.datetime,'reason':'entry_cutoff'});continue
        if len(active)>=3:
            skipped.append({'method':method,'signal_time':signal.datetime,'reason':'method_max_3'});continue
        strike=float(signal.entry_strike)
        call=quotes.get((minute,strike,'CE'));put=quotes.get((minute,strike,'PE'))
        if call is None or put is None:
            skipped.append({'method':method,'signal_time':signal.datetime,'reason':'missing_next_minute_quote'});continue
        active.append({'method':method,'signal_time':signal.datetime,'entry_time':minute,
                       'strike':strike,'entry_call_bid':call[0],
                       'entry_put_bid':put[0],'entry_premium':call[0]+put[0]})
pd.DataFrame(trades).to_csv(OUT/'paper_trades_by_method.csv',index=False)
pd.DataFrame(skipped).to_csv(OUT/'paper_skipped_by_method.csv',index=False)
report=[]
for method in methods:
    t=[v for v in trades if v['method']==method]
    report.append({'method':method,'predictions':int((p.paper_route==method).sum()),
                   'qualifying_signals':len(signals[method]),'trades_closed':len(t),
                   'skipped':sum(v['method']==method for v in skipped),
                   'gross_profit':sum(max(v['gross_pnl'],0) for v in t),
                   'gross_loss':sum(min(v['gross_pnl'],0) for v in t),
                   'net_gross_pnl':sum(v['gross_pnl'] for v in t),
                   'still_open':len(books[method])})
(OUT/'three_method_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
