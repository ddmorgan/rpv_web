from __future__ import annotations

import json
import mimetypes
import os
import warnings
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

warnings.filterwarnings("ignore", category=DeprecationWarning)
import cgi

from predictors import available_models, dataframe_to_csv, predict, read_input_file


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
EXAMPLE_ROOT = ROOT / "examples"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


class RPVHandler(BaseHTTPRequestHandler):
    server_version = "RPVPredictor/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/health":
            self._send_json({"ok": True})
            return

        if path == "/api/models":
            self._send_json({"models": available_models()})
            return

        if path == "/":
            self._send_file(STATIC_ROOT / "index.html")
            return

        if path.startswith("/examples/"):
            self._send_file_safely(EXAMPLE_ROOT, path.replace("/examples/", "", 1))
            return

        if path.startswith("/static/"):
            self._send_file_safely(STATIC_ROOT, path.replace("/static/", "", 1))
            return

        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self._send_error(HTTPStatus.BAD_REQUEST, "Expected multipart/form-data.")
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )

            upload = form["file"] if "file" in form else None
            if upload is None or not getattr(upload, "filename", ""):
                self._send_error(HTTPStatus.BAD_REQUEST, "Upload a CSV, XLSX, or JSON file.")
                return

            selected_models = _parse_models(form.getvalue("models"))
            filename = Path(upload.filename).name
            payload = upload.file.read()
            df = read_input_file(filename, payload)
            result = predict(df, selected_models=selected_models)

            accept = self.headers.get("Accept", "")
            if "text/csv" in accept:
                self._send_text(dataframe_to_csv(result["results"]), content_type="text/csv")
            else:
                self._send_json(result)
        except Exception as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def _send_file_safely(self, root: Path, relative_path: str) -> None:
        candidate = (root / relative_path).resolve()
        if not str(candidate).startswith(str(root.resolve())):
            self._send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        self._send_file(candidate)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, data: str, content_type: str = "text/plain") -> None:
        encoded = data.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


def _parse_models(raw: object) -> list[str] | None:
    if raw is None:
        return None
    values = raw if isinstance(raw, list) else [raw]
    models: list[str] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip().upper()
            if item:
                models.append(item)
    return models or None


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), RPVHandler)
    print(f"RPV predictor running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
