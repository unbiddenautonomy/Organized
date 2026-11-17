"""Separate dashboard preview; can save cutoffs/run replays but starts no live engine."""
from pathlib import Path
import sys
import argparse
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'Inference_full_2026-10-24/Inference'))
from live.live_inference import DashboardServer

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8782);args=p.parse_args()
    view=SimpleNamespace(status=lambda:{'runtime':{'mode':'43Weeks_preview_only'},'state':'PREVIEW'},recent=lambda limit:[])
    server=DashboardServer('127.0.0.1',args.port,ROOT.parent/'Inference_full_2026-10-24/Inference/live/live_dashboard.html',view)
    print(f'Preview: http://127.0.0.1:{args.port}/firstTwoWeeks',flush=True)
    try: server.server.serve_forever()
    except KeyboardInterrupt: server.server.server_close()
