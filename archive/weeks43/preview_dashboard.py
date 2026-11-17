"""Compatibility launcher: old bookmarks redirect to the actual live service."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'ivrv/engine/Inference'))
from live.dashboard_redirect import DashboardRedirect

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8782);args=p.parse_args()
    server=DashboardRedirect(args.port,8780)
    print(f'Live dashboard redirect: http://127.0.0.1:{args.port}/',flush=True)
    try: server.server.serve_forever()
    except KeyboardInterrupt: server.server.server_close()
