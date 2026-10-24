"""Local UI. Run through start.ps1 or .venv/Scripts/python.exe app.py."""
import argparse
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr
from model_host.controller import Controller
from model_host.chat_ui import build_chat

controller = None


def controls():
    gr.Markdown("### Kaggle session")
    hours = gr.Slider(0.25, 12, value=3, step=0.25, label="Session hours (quota applies)")
    start = gr.Button("Start notebook", variant="primary")
    stop = gr.Button("Stop notebook", variant="stop")
    refresh = gr.Button("Refresh status")
    stage = gr.Markdown(lambda: "**Host:** " + controller.report.get("stage", "Not started"))
    with gr.Accordion("Notebook details", open=False):
        status = gr.JSON(label="Host status", value=lambda: controller.status(refresh=False))
    info = gr.Markdown("Start notebook allocates a new GPU session. Keep this local app running.")
    start.click(controller.start, hours, info, api_name=False, concurrency_limit=1)
    stop.click(controller.stop, outputs=info, queue=False, api_name=False)
    refresh.click(controller.status, outputs=status, queue=False, api_name=False)
    gr.Timer(15).tick(lambda: controller.status(refresh=False), outputs=status, queue=False, api_name=False)
    gr.Timer(5).tick(lambda: "**Host:** " + controller.report.get("stage", "Not started"), outputs=stage, queue=False, api_name=False)
    with gr.Accordion("Share chat with other users", open=False):
        gr.Markdown("Users share one inference slot and wait in the queue. The URL expires with the Kaggle session.")
        link = gr.Textbox(label="Gradio URL", interactive=False)
        username = gr.Textbox(label="Username", interactive=False)
        password = gr.Textbox(label="Password", type="password", interactive=False)
        gr.Button("Get shared chat login").click(
            lambda: (controller.host_auth.get("url", "Not ready"),
                     controller.host_auth.get("username", ""), controller.host_auth.get("password", "")),
            outputs=[link, username, password], queue=False, api_name=False)


def main():
    global controller
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true", help="Submit and run Kaggle notebook immediately")
    parser.add_argument("--share", action="store_true", help="Share the local chat UI with authenticated users")
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    import requests
    try:
        response = requests.get("http://127.0.0.1:7861/config", timeout=1)
        existing = response.json() if response.ok else {}
        if existing.get("title") == "DeepSeek R1 · Local Chat":
            print("Local UI already running: http://127.0.0.1:7861. Use Start notebook in that window.")
            if args.open_browser:
                import webbrowser
                webbrowser.open("http://127.0.0.1:7861")
            return
    except (requests.RequestException, ValueError):
        pass
    controller = Controller()
    demo = build_chat(controller.stream, title="DeepSeek R1 · Local Chat", controls=controls)
    auth = (controller.auth["username"], controller.auth["password"]) if args.share else None
    demo.launch(server_name="127.0.0.1", server_port=7861, share=args.share, auth=auth,
                prevent_thread_lock=True, inbrowser=args.open_browser, quiet=True, mcp_server=False)
    if os.getenv("HOST_DEBUG_QUEUE") == "1":
        import threading
        import time
        import json
        def queue_diagnostics():
            while True:
                q = demo._queue
                snapshot = {"stopped": q.stopped, "worker_slots": len(q.active_jobs),
                    "active": [[e.fn.name for e in job] if job else None for job in q.active_jobs],
                    "queues": {key: {"active": value.current_concurrency,
                        "limit": value.concurrency_limit, "waiting": [e.fn.name for e in value.queue]}
                        for key, value in q.event_queue_per_concurrency_id.items()}}
                (controller.state / "local-queue.json").write_text(json.dumps(snapshot), encoding="utf-8")
                time.sleep(5)
        threading.Thread(target=queue_diagnostics, daemon=True).start()
    print("Local UI: http://127.0.0.1:7861", flush=True)
    print("Credentials are stored locally in .state/access.json (not in the notebook).", flush=True)
    if args.start:
        try:
            print(controller.start(args.hours), flush=True)
        except Exception as exc:
            print("Kaggle launch failed:", str(exc), flush=True)
    demo.block_thread()


if __name__ == "__main__":
    main()
