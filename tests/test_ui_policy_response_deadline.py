"""A slow valid local policy answer must outlive the former 10-second cutoff."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from test_ui_intent_choice import subgoal
from test_ui_observation_boundary import Adapter

from portal.computer_use import ComputerUseStore, SystemOneUiProvider


def test_delayed_valid_policy_reply_is_consumed_once_before_guarded_input():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            criteria = body['questions']['action']['criteria']
            choice = next(k for k, v in criteria.items() if json.loads(v).get('verb') == 'double_click')
            # Real HTTP delay crosses the old production read deadline. The
            # resulting input still uses the ordinary target/freshness guards.
            time.sleep(10.25)
            reply = json.dumps({'model': body['model'], 'answers': {'action': {'choice': choice,
                'probabilities': {k: float(k == choice) for k in criteria}}}}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(reply)))
            self.end_headers()
            try:
                self.wfile.write(reply)
            except BrokenPipeError:
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    provider = SystemOneUiProvider(f'http://127.0.0.1:{server.server_port}', mode='active')
    adapter = Adapter()
    store = ComputerUseStore(None, provider=provider, adapter_factory=lambda *_: adapter)
    cancel = store.begin_task('session', private_boundaries=True)
    try:
        with store.task_scope('session', 'desktop', cancel):
            result = store.intent_boundary('session', subgoal(), goal='Open the observed input folder')
        assert len(requests) == len(adapter.calls) == 1
        assert result['policy_receipts'][0]['model_confirmed'] is True
        assert result['status'] == 'observed_unverified' and result['task_progress'] is False
    finally:
        provider.client.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
