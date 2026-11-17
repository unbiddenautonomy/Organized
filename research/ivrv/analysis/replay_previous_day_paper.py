"""Conservative minute-resolution paper replay of the IV/RV routed signals."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis_outputs'/'previous_day_ivrv_2026-10-29'
p=pd.read_csv(OUT/'predictions.csv',parse_dates=['datetime','expiry'])
o=pd.read_csv(ROOT/'Data'/'banknifty_atm5_minute.csv')
o['minute']=pd.to_datetime(o.minute)
quotes={(r.minute,float(r.strike),r.option_type):(float(r.BestBuyPrice_1),float(r.BestSellPrice_1)) for r in o.itertuples()}
signals={(r.datetime+pd.Timedelta(minutes=1)):r for r in p.itertuples() if r.paper_signal and r.quality_entry_ok}
active=[]; trades=[]; skipped=[]
for minute in sorted(o.minute.unique()):
    # Mark existing positions on the current minute's ask; target/stop take precedence.
    remaining=[]
    for pos in active:
        call=quotes.get((minute,pos['strike'],'CE'))
        put=quotes.get((minute,pos['strike'],'PE'))
        if call is None or put is None:
            remaining.append(pos);continue
        exit_premium=call[1]+put[1]
        pnl=(pos['entry_premium']-exit_premium)*30
        elapsed=int((minute-pos['entry_time']).total_seconds()/60)
        reason=('target' if pnl>=1500 else 'stop' if pnl<=-750 else
                'horizon' if elapsed>=120 else 'square_off' if minute.time()>=pd.Timestamp('15:14:30').time() else None)
        if reason:
            trades.append({**pos,'exit_time':minute,'exit_call_ask':call[1],
                           'exit_put_ask':put[1],'gross_pnl':pnl,'reason':reason})
        else:remaining.append(pos)
    active=remaining
    signal=signals.get(minute)
    if signal is None:continue
    if minute.time()>=pd.Timestamp('15:14:30').time():
        skipped.append({'signal_time':signal.datetime,'reason':'entry_cutoff'});continue
    if len(active)>=3:
        skipped.append({'signal_time':signal.datetime,'reason':'max_open_positions'});continue
    strike=float(signal.entry_strike)
    call=quotes.get((minute,strike,'CE'));put=quotes.get((minute,strike,'PE'))
    if call is None or put is None:
        skipped.append({'signal_time':signal.datetime,'reason':'missing_next_minute_quote'});continue
    active.append({'signal_time':signal.datetime,'entry_time':minute,'strike':strike,
                   'route':signal.paper_route,'entry_call_bid':call[0],
                   'entry_put_bid':put[0],'entry_premium':call[0]+put[0]})
pd.DataFrame(trades).to_csv(OUT/'paper_trades.csv',index=False)
pd.DataFrame(skipped).to_csv(OUT/'paper_skipped.csv',index=False)
summary=json.loads((OUT/'run_summary.json').read_text())
summary['paper_replay']={'closed_trades':len(trades),'skipped_signals':len(skipped),
                         'still_open_at_data_end':len(active),
                         'gross_pnl_closed':sum(t['gross_pnl'] for t in trades),
                         'exit_reasons':pd.Series([t['reason'] for t in trades]).value_counts().to_dict(),
                         'entry_timing':'next minute after signal; minute bid',
                         'exit_timing':'minute ask; target 1500, stop -750, 120m horizon, square-off'}
(OUT/'run_summary.json').write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary['paper_replay'],indent=2,default=str))
