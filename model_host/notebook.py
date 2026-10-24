"""Generate self-contained, reviewable Kaggle notebooks from the tested sources."""
from pathlib import Path
import json
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def create(folder, owner, config=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(config or {})
    runtime_source = (ROOT / "model_host/runtime.py").read_text(encoding="utf-8")
    ui_source = (ROOT / "model_host/chat_ui.py").read_text(encoding="utf-8")
    setup = ("import types, sys, json\n"
             "runtime = types.ModuleType('deepseek_runtime')\n"
             "sys.modules[runtime.__name__] = runtime\n"
             f"exec({runtime_source!r}, runtime.__dict__)\n"
             f"runtime.CONFIG.update(json.loads({json.dumps(config)!r}))\n"
             "runtime.hardware()")
    ui = ("ui = types.ModuleType('deepseek_ui')\n"
          f"exec({ui_source!r}, ui.__dict__)\n"
          "demo = ui.launch_host(runtime)")
    sections = [
        ("1. Hardware check", setup),
        ("2. Build llama.cpp", "runtime.build()"),
        ("3. Download model", "runtime.download()"),
        ("4. Start server", "runtime.STOP.clear()\nruntime.start()"),
        ("5. Server validation", "runtime.validate()"),
        ("6. Gradio UI", ui),
        ("7. Benchmark", "runtime.benchmark()\n# Keep batch execution alive; Stop notebook releases the GPU VM.\n"
         "if runtime.CONFIG.get('batch_mode', False):\n    runtime.keep_alive()"),
        ("8. Shutdown / cleanup", "# Safe to run manually. Scratch model is kept for reruns within this VM.\n"
         "runtime.STOP.set()\nruntime.shutdown()\nprint(json.dumps(runtime.summary(), indent=2))"),
    ]
    cells = [nb.v4.new_markdown_cell("# DeepSeek R1 Distill Qwen 32B — pure inference\n"
             "T4 ×2 / Q4_K_M. No system prompt, tools, retrieval, or agent. "
             "MODE A inline; MODE B authenticated temporary Gradio share. "
             "Run cells 1–7 to chat; cell 8 explicitly stops the server. "
             "With batch_mode enabled, cell 7 waits for Stop notebook or the session deadline.")]
    for title, code in sections:
        if title.startswith("1."):
            code = code.replace("runtime.hardware()", "runtime.hardware()\nruntime.notify('Hardware check passed')")
        if not title.startswith("8."):
            code = ("try:\n" + "\n".join("    " + line for line in code.splitlines())
                    + "\nexcept Exception as exc:\n"
                    + "    import traceback\n"
                    + "    traceback.print_exc()\n"
                    + "    runtime.REPORT.update(error=str(exc), stage='Failed', run_id=runtime.CONFIG.get('run_id'))\n"
                    + "    runtime.save()\n"
                    + "    print('HOST_PROGRESS:' + json.dumps(runtime.REPORT), flush=True)\n"
                    + "    if runtime.CONFIG.get('binary'):\n"
                    + "        runtime.shutdown_server()\n"
                    + "    raise\n")
        cells += [nb.v4.new_markdown_cell("## " + title), nb.v4.new_code_cell(code)]
    notebook = nb.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    nb.validate(notebook)
    nb.write(notebook, folder / "deepseek.ipynb")
    metadata = {"id": f"{owner}/deepseek-r1-32b-local-chat", "title": "Deepseek R1 32b Local Chat",
                "code_file": "deepseek.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata["id"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a standalone Kaggle notebook")
    parser.add_argument("--owner", default="your-kaggle-username", help="Your Kaggle username")
    args = parser.parse_args()
    create(ROOT / "notebook", args.owner, {"mode": "A", "batch_mode": False})
