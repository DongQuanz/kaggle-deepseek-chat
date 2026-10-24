"""Authenticated Kaggle control and remote Gradio inference."""
from __future__ import annotations
import asyncio
import base64
import json
import os
import re
import secrets
import threading
import time
from gradio_client import Client
from .notebook import create, ROOT


class Controller:
    def __init__(self, monitor=True):
        self.lock = threading.RLock()
        self.state = ROOT / ".state"
        self.state.mkdir(exist_ok=True)
        credentials = self.state / "access.json"
        if credentials.exists():
            self.auth = json.loads(credentials.read_text())
        else:
            self.auth = {"username": os.getenv("GRADIO_USER", "model-user"),
                         "password": os.getenv("GRADIO_PASSWORD") or secrets.token_urlsafe(24)}
            credentials.write_text(json.dumps(self.auth), encoding="utf-8")
        self.report = {}
        self.host_auth = {}
        self.kernel = None
        self.run_id = None
        self.stop_requested = False
        self.starting = False
        self.last_contact = 0
        self.last_access = None
        self.monitor_stop = threading.Event()
        self.finished = False
        self.log_thread = None
        session = self.state / "session.json"
        if session.exists():
            saved = json.loads(session.read_text())
            self.kernel = saved.get("kernel")
            self.run_id = saved.get("run_id")
            self.report = saved.get("report", {})
            self.stop_requested = saved.get("stop_requested", False)
        host_file = self.state / "host-access.json"
        if host_file.exists():
            saved = json.loads(host_file.read_text())
            if saved.get("run_id") == self.run_id:
                self.host_auth = saved
        if monitor:
            threading.Thread(target=self.watch, daemon=True, name="kaggle-status").start()

    def persist(self):
        with self.lock:
            temporary = self.state / ("session-" + secrets.token_hex(6) + ".tmp")
            payload = json.dumps({"kernel": self.kernel, "run_id": self.run_id,
                "report": self.report, "stop_requested": self.stop_requested}, indent=2)
            try:
                temporary.write_text(payload, encoding="utf-8")
                for attempt in range(4):
                    try:
                        temporary.replace(self.state / "session.json")
                        self.report.pop("state_save_error", None)
                        return
                    except PermissionError:
                        time.sleep(.05 * (attempt + 1))
                # Windows readers may permit writing but temporarily deny file replacement.
                (self.state / "session.json").write_text(payload, encoding="utf-8")
            except OSError as exc:
                self.report["state_save_error"] = str(exc)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def api():
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api

    def kaggle_status(self):
        if not self.kernel:
            return {"status": "not_started"}
        result = self.api().kernels_status(self.kernel)
        return result.to_dict() if hasattr(result, "to_dict") else dict(result)

    @staticmethod
    def active(status):
        value = str(status.get("status", "")).lower()
        return any(x in value for x in ("running", "queued", "pending"))

    def public_key(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        path = self.state / "private-key.pem"
        if not path.exists():
            key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
            path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        return key.public_key().public_bytes(serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo).decode()

    def ingest(self, log, write=True):
        try:
            entries = json.loads(log or "[]")
            text = "".join(entry.get("data", "") for entry in entries)
        except (ValueError, TypeError):
            text = log or ""
        if write and text:
            (self.state / "kaggle.log").write_text(text, encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("HOST_PROGRESS:"):
                try:
                    report = json.loads(line[len("HOST_PROGRESS:"):])
                    if report.get("run_id") == self.run_id:
                        self.report.update(report)
                except ValueError:
                    continue
            if line.startswith("HOST_ACCESS:"):
                encrypted = line[len("HOST_ACCESS:"):].strip()
                if encrypted == self.last_access:
                    continue
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import padding
                key = serialization.load_pem_private_key((self.state / "private-key.pem").read_bytes(), password=None)
                access = json.loads(key.decrypt(base64.b64decode(encrypted), padding.OAEP(
                    mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)))
                if access.get("run_id") != self.run_id:
                    continue
                from urllib.parse import urlparse
                url = urlparse(access["url"])
                if url.scheme != "https" or not (url.hostname or "").endswith(".gradio.live"):
                    raise ValueError("Expected HTTPS gradio.live host URL")
                self.host_auth = access
                self.report["gradio_url"] = access["url"]
                (self.state / "host-access.json").write_text(json.dumps(access), encoding="utf-8")
                self.last_access = encrypted
        self.persist()

    def poll(self):
        if not self.kernel:
            return
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        status = self.kaggle_status()
        self.report["kaggle"] = status
        owner, slug = self.kernel.split("/")
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.page_size = 1
        with self.api().build_kaggle_client() as api:
            response = api.kernels.kernels_api_client.list_kernel_session_output(request)
        self.ingest(response.log)
        self.last_contact = time.monotonic()
        self.report.pop("monitor_error", None)
        value = str(status.get("status", "")).lower()
        if any(word in value for word in ("error", "fail", "cancel")):
            self.report["stage"] = "Kaggle run failed: " + str(status.get("failureMessage") or status)
            self.finished = True
        elif "complete" in value:
            self.report["stage"] = "Kaggle run completed; GPUs released"
            self.finished = True
        if self.stop_requested and self.host_auth and not self.finished and not self.report.get("stopped"):
            self.stop()
        self.persist()

    def watch(self):
        while not self.monitor_stop.wait(15):
            if self.kernel and not self.starting and not self.finished:
                try:
                    if not self.log_thread or not self.log_thread.is_alive():
                        self.log_thread = threading.Thread(target=self.live_logs,
                            args=(self.kernel, self.run_id), daemon=True, name="kaggle-live-logs")
                        self.log_thread.start()
                    self.poll()
                except Exception as exc:
                    self.report["monitor_error"] = str(exc)[:2000]

    def live_logs(self, kernel, run_id):
        pending = ""
        try:
            with (self.state / "kaggle-live.log").open("a", encoding="utf-8") as output:
                for event in self.api().kernels_logs_stream(kernel):
                    if self.run_id != run_id or self.monitor_stop.is_set():
                        return
                    data = event.get("data", "")
                    self.report.pop("live_log_error", None)
                    output.write(data)
                    output.flush()
                    pending += data
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        if line.startswith(("HOST_PROGRESS:", "HOST_ACCESS:")):
                            self.ingest(line, write=False)
                        percent = re.match(r"\[\s*(\d+)%\]", line)
                        if percent and "Building llama.cpp" in self.report.get("stage", ""):
                            self.report["stage"] = f"Building llama.cpp with CUDA ({percent.group(1)}%)"
                            self.persist()
        except Exception as exc:
            self.report["live_log_error"] = str(exc)[:1500]

    def start(self, hours=3):
        with self.lock:
            if self.starting:
                return "A launch is already in progress."
            self.starting = True
        try:
            if self.kernel:
                status = self.kaggle_status()
                if self.active(status):
                    return "Existing Kaggle run is active; duplicate GPU session was not started."
            api = self.api()
            owner = api.config_values.get("username") or os.getenv("KAGGLE_USERNAME")
            if not owner:
                raise RuntimeError("Set KAGGLE_USERNAME to your Kaggle account slug.")
            hours = max(.25, min(float(hours), 12))
            self.run_id = secrets.token_hex(12)
            self.stop_requested = False
            self.finished = False
            self.host_auth = {}
            self.last_access = None
            self.report = {"stage": "Submitting Kaggle notebook", "run_id": self.run_id}
            (self.state / "kaggle-live.log").write_text("", encoding="utf-8")
            folder = ROOT / ".runs" / "current"
            self.kernel = create(folder, owner, {"public_key": self.public_key(), "run_id": self.run_id,
                "mode": "B", "batch_mode": True, "session_hours": hours})
            self.persist()
            result = api.kernels_push(str(folder), timeout=int((hours + .5) * 3600), acc="NvidiaTeslaT4")
            error = getattr(result, "error", None)
            if error:
                raise RuntimeError(str(error))
            self.report["stage"] = "Submitted; waiting for Kaggle GPU allocation / hardware check"
            self.persist()
            return "Notebook submitted: https://www.kaggle.com/code/" + self.kernel
        except Exception as exc:
            self.report["stage"] = "Launch failed: " + str(exc)[:2000]
            self.persist()
            raise
        finally:
            self.starting = False

    def client(self):
        if not self.host_auth:
            raise RuntimeError("Kaggle is not ready. Check notebook status and progress.")
        if self.finished or self.report.get("stopped"):
            raise RuntimeError("Kaggle session has ended. Start notebook to allocate a new session.")
        return Client(self.host_auth["url"], auth=(self.host_auth["username"], self.host_auth["password"]),
                      verbose=False, httpx_kwargs={"timeout": 30})

    def stop(self):
        self.stop_requested = True
        self.persist()
        if self.report.get("stopped") or self.finished:
            return "Notebook has stopped."
        if self.host_auth:
            client = self.client()
            try:
                result = client.predict(self.host_auth["admin_token"], api_name="/shutdown")
                return "Stop acknowledged. The notebook will release GPUs within about 15 seconds. " + str(result)
            finally:
                client.close()
        return ("Stop recorded; it will be sent as soon as the host API is ready. "
                "Kaggle's documented API has no cancel command for queued/building runs. "
                "The configured timeout remains in effect.")

    def status(self, refresh=True):
        if refresh and self.kernel:
            try:
                self.poll()
            except Exception as exc:
                self.report["monitor_error"] = str(exc)[:1500]
        return dict(self.report, kernel=self.kernel, stop_requested=self.stop_requested)

    async def stream(self, messages, temperature, top_p, maximum, behavior):
        client = await asyncio.to_thread(self.client)
        job = client.submit(messages, temperature, top_p, maximum, behavior, api_name="/infer")
        sentinel = object()
        iterator = iter(job)
        def next_item():
            return next(iterator, sentinel)
        try:
            while True:
                result = await asyncio.to_thread(next_item)
                if result is sentinel:
                    break
                yield result
            await asyncio.to_thread(job.result)
        finally:
            job.cancel()
            client.close()
