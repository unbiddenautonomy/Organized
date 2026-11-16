"""Keep old dashboard bookmarks pointed at the actual service, never a preview."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,urlunsplit
import threading


class DashboardRedirect:
    def __init__(self,port=8782,target_port=8780):
        if port==target_port:raise ValueError('Dashboard redirect must use a different port')
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path=urlsplit(self.path)
                target=urlunsplit(('http',f'127.0.0.1:{int(target_port)}',path.path or '/',path.query,''))
                self.send_response(307)
                self.send_header('Location',target)
                self.send_header('Cache-Control','no-store')
                self.send_header('Content-Length','0')
                self.end_headers()
            do_HEAD=do_GET
            def log_message(self,*args):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',int(port)),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,name='dashboard-bookmark-redirect',daemon=True)
    def start(self):self.thread.start()
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
