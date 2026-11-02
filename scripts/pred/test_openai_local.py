"""Regression test for the RULER local OpenAI-compatible client fork."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from client_wrappers import OpenAIClient


def test_openai_local_without_azure_credentials(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            response = json.dumps({
                'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'local-model', 'choices': [{'index': 0,
                    'message': {'role': 'assistant', 'content': '42'},
                    'finish_reason': 'stop'}],
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        monkeypatch.setenv('OPENAI_BASE_URL', f'http://127.0.0.1:{server.server_port}/v1')
        for key in ('OPENAI_API_KEY', 'AZURE_API_ID', 'AZURE_API_SECRET', 'AZURE_API_ENDPOINT'):
            monkeypatch.delenv(key, raising=False)
        client = OpenAIClient('local-model', tokens_to_generate=5, temperature=0,
                              random_seed=1, top_p=1, stop=[])
        assert client('answer?') == {'text': ['42']}
        assert client.process_batch(['a', 'b']) == [{'text': ['42']}, {'text': ['42']}]
        assert requests[0][0] == '/v1/chat/completions'
        assert requests[0][1]['model'] == 'local-model'
        assert requests[0][1]['messages'][0]['content'] == 'answer?'
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
