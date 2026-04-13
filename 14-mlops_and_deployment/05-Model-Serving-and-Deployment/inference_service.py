from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer


@dataclass
class DemoModel:
    weights: list[float]
    bias: float

    def predict_score(self, features: list[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features)) + self.bias

    def predict_label(self, features: list[float]) -> str:
        return "positive" if self.predict_score(features) >= 0.5 else "negative"


MODEL = DemoModel(weights=[0.4, 0.2, 0.1], bias=0.05)


class InferenceHandler(BaseHTTPRequestHandler):
    def _write_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json({"status": "ok", "model": "demo_linear_classifier"})
            return
        self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        features = payload.get("features", [])
        if not isinstance(features, list) or len(features) != 3:
            self._write_json({"error": "features must be a list of length 3"}, status=HTTPStatus.BAD_REQUEST)
            return
        score = MODEL.predict_score([float(value) for value in features])
        label = MODEL.predict_label([float(value) for value in features])
        self._write_json({"score": round(score, 4), "label": label})


def run_server(host: str, port: int) -> None:
    server = HTTPServer((host, port), InferenceHandler)
    print(f"Serving on http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny inference service demo.")
    parser.add_argument("--serve", action="store_true", help="Start the HTTP server instead of printing an example prediction.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.serve:
        run_server(args.host, args.port)
    else:
        features = [0.9, 0.3, 0.7]
        print({"features": features, "score": round(MODEL.predict_score(features), 4), "label": MODEL.predict_label(features)})


if __name__ == "__main__":
    main()
