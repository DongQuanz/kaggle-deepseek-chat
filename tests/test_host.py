import asyncio
import json
import threading
import time
from pathlib import Path
import httpx
import nbformat
import pytest
from model_host.chat_ui import clean_messages, prepare_request, make_llama_stream, build_chat, add_host_api
from model_host.notebook import create


def test_rejects_system_tools_and_non_text():
    for message in ({"role": "system", "content": "injected"},
                    {"role": "tool", "content": "output"},
                    {"role": "user", "content": ["attachment"]}):
        with pytest.raises(ValueError):
            clean_messages([message])


def test_context_limits_and_removes_complete_old_turns():
    async def check():
        def handler(request):
            payload = json.loads(request.content)
            if request.url.path == "/apply-template":
                return httpx.Response(200, json={"prompt": json.dumps(payload["messages"])})
            turns = json.loads(payload["content"])
            return httpx.Response(200, json={"tokens": list(range(len(turns) * 100))})
        messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": "latest"}]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            sent, limit, note = await prepare_request(client, "http://test", messages, 400, 8192, "Keep recent turns")
            assert sent == [messages[-1]]
            assert limit == 292
            assert "omitted 2" in note
            sent, limit, _ = await prepare_request(client, "http://test", messages, 400, 8192, "Keep all / reject overflow")
            assert sent == messages and limit == 92
            with pytest.raises(ValueError, match="exceeds context"):
                await prepare_request(client, "http://test", messages, 200, 8192, "Keep all / reject overflow")
    asyncio.run(check())


def test_notebook_cells_and_credentials(tmp_path):
    create(tmp_path, "example", {"mode": "A"})
    notebook = nbformat.read(tmp_path / "deepseek.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [c.source for c in notebook.cells if c.cell_type == "code"]
    assert len(code) == 8
    for cell in code:
        compile(cell, "notebook", "exec")
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["is_private"] is True
    assert "helpful assistant" not in "".join(code)
    assert "runtime.keep_alive()" in code[6]


def test_real_gradio_authenticated_stream_and_stop(tmp_path, monkeypatch):
    from gradio_client import Client
    released = threading.Event()
    async def fake_stream(messages, temp, top_p, max_tokens, behavior):
        try:
            for i in range(3):
                yield {"messages": messages + [{"role": "assistant", "content": f"token {i}"}],
                       "elapsed": i / 100, "usage": {"completion_tokens": i}, "timings": {}}
                await asyncio.sleep(30 if messages[-1]["content"] == "slow" else .01)
        finally:
            if messages[-1]["content"] == "slow":
                released.set()
    stopped = threading.Event()
    def stop(token):
        if token != "test-admin":
            raise ValueError("bad token")
        stopped.set()
        return {"stopping": True}
    demo = build_chat(fake_stream, title="Integration test")
    add_host_api(demo, fake_stream, lambda: {"ready": True}, stop)
    try:
        _, url, _ = demo.launch(server_name="127.0.0.1", server_port=7875, share=False,
                                auth=("tester", "temporary-test-password"), prevent_thread_lock=True, quiet=True)
        with pytest.raises(Exception):
            Client(url, verbose=False)
        client = Client(url, auth=("tester", "temporary-test-password"), verbose=False)
        assert client.predict(api_name="/host_status") == {"ready": True}
        job = client.submit([{"role": "user", "content": "Hello"}], .6, .95, 8192,
                            "Keep recent turns", api_name="/infer")
        outputs = list(job)
        assert outputs[-1]["messages"][-1]["content"] == "token 2"
        from model_host import controller as module
        monkeypatch.setattr(module, "ROOT", tmp_path)
        local = module.Controller(monitor=False)
        local.host_auth = {"url": url, "username": "tester", "password": "temporary-test-password"}
        # Only this isolated integration fixture may connect to an HTTP test server.
        monkeypatch.setattr(local, "validate_host_url", lambda value: value)
        async def bridge():
            return [result async for result in local.stream([{"role": "user", "content": "bridge"}],
                                                           .6, .95, 32, "Keep recent turns")]
        assert asyncio.run(bridge())[-1]["messages"][-1]["content"] == "token 2"
        slow = client.submit([{"role": "user", "content": "slow"}], .6, .95, 32,
                             "Keep recent turns", api_name="/infer")
        assert next(iter(slow))["messages"][-1]["content"] == "token 0"
        slow.cancel()
        assert released.wait(10), "Cancelled generation must release the inference slot"
        assert client.predict("test-admin", api_name="/shutdown") == {"stopping": True}
        assert stopped.is_set()
        client.close()
    finally:
        demo.close()


def test_oom_fallback_order(tmp_path, monkeypatch):
    from model_host import runtime
    import requests
    commands = []
    monkeypatch.setattr(runtime, "WORK", tmp_path)
    monkeypatch.setattr(runtime, "CONFIG", {"binary": "/fake/llama-server", "model": "/scratch/model.gguf"})
    monkeypatch.setattr(runtime, "notify", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "shutdown_server", lambda: None)
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "--fit")
    class Process:
        pid = 999
        def __init__(self, args, stdout, **kwargs):
            commands.append(args)
            self.ready = len(commands) == 4
            stdout.write("CUDA error: out of memory\n" if not self.ready else "ready\n")
            stdout.flush()
        def poll(self):
            return None if self.ready else 1
    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    monkeypatch.setattr(requests, "get", lambda *a, **k: type("Response", (), {"status_code": 200})())
    runtime.STOP.clear()
    runtime.start()
    assert [c[c.index("--ctx-size") + 1] for c in commands] == ["8192", "6144", "4096", "4096"]
    assert [c[c.index("--n-gpu-layers") + 1] for c in commands] == ["all", "all", "all", "60"]
    assert all(c[c.index("--tensor-split") + 1] == "1,1" for c in commands)


def test_access_packet_encrypted_and_bound_to_run(tmp_path, monkeypatch):
    from model_host import controller as module, runtime
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    monkeypatch.setattr(module, "ROOT", tmp_path)
    c = module.Controller(monitor=False)
    c.run_id = "test-current-run"
    public = c.public_key()
    monkeypatch.setattr(runtime, "CONFIG", {"public_key": public, "run_id": c.run_id, "admin_token": "test-admin"})
    monkeypatch.setenv("GRADIO_USER", "test-user")
    monkeypatch.setenv("GRADIO_PASSWORD", "password-not-in-notebook")
    printed = []
    monkeypatch.setattr("builtins.print", lambda s, **kwargs: printed.append(s))
    runtime.publish_access("https://example-test.gradio.live")
    line = printed[-1]
    assert "password-not-in-notebook" not in line
    c.ingest(json.dumps([{"data": line + "\n"}]))
    assert c.host_auth["password"] == "password-not-in-notebook"
    assert c.host_auth["admin_token"] == "test-admin"
    c.run_id = "different-run"
    c.host_auth = {}
    c.last_access = None
    c.ingest(json.dumps([{"data": line + "\n"}]))
    assert c.host_auth == {}


def test_kaggle_inline_transport_and_mode_b_requires_password(monkeypatch):
    from types import SimpleNamespace
    from model_host import chat_ui
    captured = {}
    class Demo:
        def launch(self, **kwargs):
            captured.update(kwargs)
            return None, "http://127.0.0.1:7860", "https://inline-test.gradio.live"
        def close(self):
            pass
    monkeypatch.setattr(chat_ui, "build_chat", lambda *a, **k: Demo())
    monkeypatch.setattr(chat_ui, "add_host_api", lambda *a, **k: None)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.delenv("GRADIO_PASSWORD", raising=False)
    monkeypatch.delenv("GRADIO_MODE", raising=False)
    runtime = SimpleNamespace(BASE_URL="http://127.0.0.1:8080", CONFIG={"context": 8192, "mode": "A"},
        REPORT={}, notify=lambda *a, **k: None, publish_access=lambda *a: None)
    chat_ui.launch_host(runtime)
    assert captured["share"] is True and captured["inline"] is True
    assert captured["auth"] and len(captured["auth"][1]) >= 24
    monkeypatch.delenv("GRADIO_PASSWORD", raising=False)
    runtime.CONFIG["mode"] = "B"
    with pytest.raises(RuntimeError, match="requires GRADIO_PASSWORD"):
        chat_ui.launch_host(runtime)
