"""Shared Gradio interface for local clients and the Kaggle host."""
from __future__ import annotations
import asyncio
import copy
import json
import time
import gradio as gr


def clean_messages(messages):
    if not isinstance(messages, list) or not messages or len(messages) > 200:
        raise ValueError("Conversation must contain 1–200 messages.")
    result = []
    for msg in messages:
        if msg.get("role") not in {"user", "assistant"} or not isinstance(msg.get("content"), str):
            raise ValueError("Only plain user/assistant messages are accepted.")
        if "reasoning_content" in msg and not isinstance(msg["reasoning_content"], str):
            raise ValueError("Reasoning history must be plain text.")
        result.append({k: msg[k] for k in ("role", "content", "reasoning_content") if k in msg})
    return result


def display(messages):
    result = []
    for msg in messages:
        content = msg["content"]
        if msg.get("reasoning_content"):
            content = "<details><summary>Reasoning</summary>\n\n" + msg["reasoning_content"] + "\n\n</details>\n\n" + content
        result.append({"role": msg["role"], "content": content})
    return result


def metrics_text(data):
    usage, timing = data.get("usage") or {}, data.get("timings") or {}
    rate = timing.get("predicted_per_second")
    elapsed = data.get("elapsed", 0)
    parts = [f"Total: {elapsed:.1f} s"]
    if timing.get("predicted_ms") is not None:
        parts.append(f"Generation time: {timing['predicted_ms']/1000:.1f} s")
    if usage:
        parts += [f"Input: {usage.get('prompt_tokens', '?')} tokens", f"Output: {usage.get('completion_tokens', '?')} tokens"]
    if rate is not None:
        parts.append(f"Generation: {rate:.2f} tok/s")
    if timing.get("prompt_per_second") is not None:
        parts.append(f"Prompt: {timing['prompt_per_second']:.2f} tok/s")
    if data.get("note"):
        parts.append(data["note"])
    return " · ".join(parts)


async def prepare_request(client, base_url, messages, context, max_tokens, behavior):
    messages = clean_messages(messages)
    original_length = len(messages)
    requested = max(1, min(int(max_tokens), 8192))
    while True:
        rendered = await client.post(base_url + "/apply-template", json={"messages": messages})
        rendered.raise_for_status()
        tokenized = await client.post(base_url + "/tokenize", json={"content": rendered.json()["prompt"], "add_special": False})
        tokenized.raise_for_status()
        prompt_tokens = len(tokenized.json()["tokens"])
        available = context - prompt_tokens - 8
        reserve = min(requested, context // 2)
        if behavior == "Keep recent turns" and available < reserve and len(messages) > 2:
            messages = messages[2:]
            continue
        if available <= 0:
            raise ValueError(f"Prompt exceeds context {context}. Shorten it or start a new chat.")
        limit = min(requested, available)
        note = f"Output budget: {limit}/{requested}; context {context}"
        if len(messages) < original_length:
            note += f"; omitted {original_length-len(messages)} oldest messages for this request"
        return messages, limit, note


def make_llama_stream(base_url, context_getter):
    async def stream(messages, temperature, top_p, max_tokens, behavior):
        import httpx
        start = time.monotonic()
        messages = clean_messages(messages)
        if messages[-1]["role"] != "user":
            raise ValueError("Last message must be a user prompt.")
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=15)) as client:
            sent, limit, note = await prepare_request(client, base_url, messages, context_getter(), max_tokens, behavior)
            payload = {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "messages": sent,
                       "temperature": max(0, min(float(temperature), 2)), "top_p": max(0.01, min(float(top_p), 1)),
                       "max_tokens": limit, "stream": True, "stream_options": {"include_usage": True}}
            result = {"messages": copy.deepcopy(messages) + [{"role": "assistant", "content": "", "reasoning_content": ""}],
                      "elapsed": 0, "usage": {}, "timings": {}, "note": note}
            async with client.stream("POST", base_url + "/v1/chat/completions", json=payload) as response:
                if response.is_error:
                    raise ValueError((await response.aread()).decode()[:1500])
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line[6:].strip() == "[DONE]":
                        break
                    data = json.loads(line[6:])
                    if data.get("error"):
                        raise ValueError(str(data["error"]))
                    choices = data.get("choices") or []
                    delta = choices[0].get("delta", {}) if choices else {}
                    for key in ("content", "reasoning_content"):
                        result["messages"][-1][key] += delta.get(key) or ""
                    result["usage"] = data.get("usage") or result["usage"]
                    result["timings"] = data.get("timings") or result["timings"]
                    result["elapsed"] = time.monotonic() - start
                    yield copy.deepcopy(result)
            result["elapsed"] = time.monotonic() - start
            yield result
    return stream


def build_chat(stream, title="DeepSeek R1 · 32B", controls=None):
    with gr.Blocks(title=title, analytics_enabled=False) as demo:
        gr.Markdown(f"# {title}\nPure LLM inference · Q4_K_M · NVIDIA T4 ×2")
        history = gr.State([])
        with gr.Row():
            with gr.Column(scale=1, min_width=245):
                if controls:
                    controls()
                gr.Markdown("### Settings")
                temperature = gr.Slider(0, 2, value=0.6, step=0.05, label="Temperature")
                top_p = gr.Slider(0.05, 1, value=0.95, step=0.01, label="Top p")
                max_tokens = gr.Slider(1, 8192, value=8192, step=1, label="Max output tokens")
                behavior = gr.Dropdown(["Keep recent turns", "Keep all / reject overflow"], value="Keep recent turns", label="Context behavior")
                gr.Markdown("Input and output share the context window. The output budget is capped to remaining space.")
                new = gr.Button("New Chat")
                clear = gr.Button("Clear conversation")
            with gr.Column(scale=4):
                chat = gr.Chatbot(height="max(260px, calc(100vh - 360px))", label="Conversation", render_markdown=True,
                                  latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                                    {"left": "\\[", "right": "\\]", "display": True},
                                                    {"left": "\\(", "right": "\\)", "display": False},
                                                    {"left": "$", "right": "$", "display": False}])
                prompt = gr.Textbox(label="Message", placeholder="Nhập tin nhắn…", lines=3)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    stop = gr.Button("Stop Generation", variant="stop")
                    regen = gr.Button("Regenerate")
                metrics = gr.Markdown("Ready when the Kaggle backend is connected.")

        async def respond(text, previous, temp, p, maximum, context_behavior, regenerate=False):
            messages = copy.deepcopy(previous or [])
            if regenerate:
                if messages and messages[-1]["role"] == "assistant":
                    messages.pop()
                if not messages or messages[-1]["role"] != "user":
                    yield display(messages), messages, "", "No response to regenerate."
                    return
            else:
                if not text.strip():
                    yield display(messages), messages, text, "Enter a message."
                    return
                messages.append({"role": "user", "content": text})
            yield display(messages), messages, "", "Waiting for the inference slot…"
            try:
                async for result in stream(messages, temp, p, maximum, context_behavior):
                    yield display(result["messages"]), result["messages"], "", metrics_text(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield display(messages), messages, "", f"Error: {str(exc)[:1800]}"

        async def regenerate(previous, temp, p, maximum, context_behavior):
            async for value in respond("", previous, temp, p, maximum, context_behavior, True):
                yield value

        inputs = [prompt, history, temperature, top_p, max_tokens, behavior]
        outputs = [chat, history, prompt, metrics]
        events = [send.click(respond, inputs, outputs, api_name=False, concurrency_id="inference", concurrency_limit=1),
                  prompt.submit(respond, inputs, outputs, api_name=False, concurrency_id="inference", concurrency_limit=1),
                  regen.click(regenerate, inputs[1:], outputs, api_name=False, concurrency_id="inference", concurrency_limit=1)]
        stop.click(fn=lambda: "Generation stopped.", outputs=metrics, cancels=events, queue=False, api_name=False)
        for button in (new, clear):
            button.click(lambda: ([], [], "", "New conversation."), outputs=outputs, cancels=events, queue=False, api_name=False)
    demo.queue(max_size=32, default_concurrency_limit=1)
    return demo


def add_host_api(demo, stream, status_fn, stop_fn):
    with demo:
        messages = gr.JSON(visible=False)
        temp = gr.Number(0.6, visible=False)
        top_p = gr.Number(0.95, visible=False)
        maximum = gr.Number(8192, visible=False)
        behavior = gr.Textbox("Keep recent turns", visible=False)
        result = gr.JSON(visible=False)
        gr.Button(visible=False).click(stream, [messages, temp, top_p, maximum, behavior], result,
                                       api_name="infer", concurrency_id="inference", concurrency_limit=1)
        gr.Button(visible=False).click(status_fn, outputs=result, api_name="host_status", queue=False)
        admin = gr.Textbox(visible=False)
        gr.Button(visible=False).click(stop_fn, admin, result, api_name="shutdown", queue=False)


def launch_host(runtime):
    import os
    stream = make_llama_stream(runtime.BASE_URL, lambda: runtime.CONFIG["context"])
    demo = build_chat(stream)
    mode = os.environ.get("GRADIO_MODE", runtime.CONFIG.get("mode", "A")).upper()
    if mode not in {"A", "B"}:
        raise ValueError("GRADIO_MODE must be A or B")
    # Hosted Kaggle notebooks cannot embed a VM's localhost URL in the user's browser.
    # Gradio's documented hosted-notebook transport uses a share tunnel for the iframe.
    share = mode == "B" or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    username = os.environ.get("GRADIO_USER", "model-user")
    password = os.environ.get("GRADIO_PASSWORD")
    if share and not password:
        try:
            from kaggle_secrets import UserSecretsClient
            password = UserSecretsClient().get_secret("GRADIO_PASSWORD")
        except Exception:
            pass
    if mode == "B" and not password:
        raise RuntimeError("MODE B requires GRADIO_PASSWORD environment variable or Kaggle Secret.")
    if share and not password:
        import secrets
        password = secrets.token_urlsafe(24)
        os.environ["GRADIO_PASSWORD"] = password
        print("Inline notebook login — username:", username, "password:", password)

    def host_status():
        runtime.vram()
        return dict(runtime.REPORT, **runtime.summary())

    def stop_host(token):
        import secrets
        if not runtime.CONFIG.get("admin_token") or not secrets.compare_digest(str(token), runtime.CONFIG["admin_token"]):
            raise ValueError("Invalid admin token")
        runtime.STOP.set()
        return {"stopping": True}

    add_host_api(demo, stream, host_status, stop_host)
    old = runtime.CONFIG.get("demo")
    if old:
        old.close()
    _, local_url, share_url = demo.launch(server_name="127.0.0.1", server_port=7860, share=share,
                                          auth=(username, password) if share else None,
                                          inline=mode == "A", prevent_thread_lock=True, quiet=True,
                                          show_error=True, mcp_server=False)
    runtime.CONFIG["demo"] = demo
    if share and not share_url:
        raise RuntimeError("Gradio share tunnel failed; localhost remains private. Retry the UI cell.")
    runtime.notify("Gradio connected", gradio_url=share_url or local_url)
    runtime.publish_access(share_url or local_url)
    return demo
