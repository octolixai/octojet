"""HTTP helpers for the CUDA server tests that need no tokenizer: a threaded server on a free port and a JSON POST."""

import http.client
import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

from tensorfold.cuda import server


@contextmanager
def http_server(app):
    class QuietServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            pass
    httpd = QuietServer(("127.0.0.1", 0), server.make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def post(port, body, chat):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        route = "/v1/chat/completions" if chat else "/v1/completions"
        connection.request("POST", route, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()
