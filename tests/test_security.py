import json
import sys
from types import SimpleNamespace
import pytest
from model_host.chat_ui import clean_messages
from model_host.controller import Controller


@pytest.mark.parametrize("url", [
    "http://example.gradio.live", "https://localhost", "https://127.0.0.1",
    "https://example.gradio.live.evil.test", "https://example.gradio.live:8443",
    "https://user:password@example.gradio.live", "https://example.gradio.live/redirect",
    "https://example.gradio.live?url=http://localhost", "https://example.gradio.live/#secret",
])
def test_reject_untrusted_host_origins(url):
    with pytest.raises(ValueError):
        Controller.validate_host_url(url)


def test_valid_host_origin():
    assert Controller.validate_host_url("https://example.gradio.live/")


def test_payload_limits_and_types():
    for value in ([None], [42], [{"role": "user", "content": "x" * (1024 * 1024 + 1)}],
                  [{"role": "user", "content": "x", "reasoning_content": {}}]):
        with pytest.raises(ValueError):
            clean_messages(value)
    result = clean_messages([{"role": "user", "content": "hi", "tools": [{"name": "execute"}]}])
    assert result == [{"role": "user", "content": "hi"}]


def test_console_authentication_is_required_without_share(tmp_path, monkeypatch):
    import requests
    import app
    captured = {}
    class Demo:
        def launch(self, **kwargs):
            captured.update(kwargs)
        def block_thread(self):
            pass
    console = SimpleNamespace(auth={"username": "owner", "password": "test-only-password"},
                              state=tmp_path / ".state", stream=None)
    monkeypatch.setattr(app, "Controller", lambda: console)
    monkeypatch.setattr(app, "build_chat", lambda *args, **kwargs: Demo())
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError()
    monkeypatch.setattr(requests, "get", unavailable)
    monkeypatch.setattr(sys, "argv", ["app.py"])
    monkeypatch.delenv("HOST_DEBUG_QUEUE", raising=False)
    app.main()
    assert captured["auth"] == ("owner", "test-only-password")
    assert captured["share"] is False and captured["server_name"] == "127.0.0.1"
    assert captured["strict_cors"] is True and captured["mcp_server"] is False
    assert str(console.state) in captured["blocked_paths"]


def test_posix_secret_permissions(tmp_path, monkeypatch):
    import os
    if os.name == "nt":
        pytest.skip("POSIX permission bits do not enforce Windows ACLs")
    from model_host import controller
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    c = controller.Controller(monitor=False)
    c.public_key()
    assert c.state.stat().st_mode & 0o777 == 0o700
    for name in ("access.json", "private-key.pem"):
        assert (c.state / name).stat().st_mode & 0o777 == 0o600
