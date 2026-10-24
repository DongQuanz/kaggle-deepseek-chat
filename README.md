# Kaggle DeepSeek Chat

A local Gradio chat interface for **DeepSeek-R1-Distill-Qwen-32B**, hosted on a private Kaggle notebook with **two NVIDIA T4 GPUs**. Start the notebook, monitor progress, chat, and request shutdown without copying URLs from Kaggle.

Uses llama.cpp with CUDA and exactly `bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf`. This is quantized inference, not LoRA/QLoRA training or full-precision inference. No system prompt is added; the GGUF chat template is used. The model has no browsing, retrieval, MCP, execution agent, or external tools.

## Requirements

- Python 3.11+ and Git installed locally. Tested on Windows with Python 3.14.
- Your own Kaggle account, API credentials, GPU access, Internet access for notebooks, and remaining GPU quota.
- T4 x2 availability. The notebook refuses other hardware.

The model is downloaded inside Kaggle, not to your local computer. Cold starts can take tens of minutes: compiling CUDA and downloading about 20 GB are required. GPU availability and session duration depend on Kaggle. Stopping a session may discard its scratch storage, so the next start may repeat setup.

## Setup

1. Clone this repository or download and extract its ZIP.
2. Create API credentials in your Kaggle account settings. For username/key credentials, save `kaggle.json` outside this repository at `%USERPROFILE%\.kaggle\kaggle.json` on Windows or `~/.kaggle/kaggle.json` on Linux/macOS. Alternatively use `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables. For token authentication, use `KAGGLE_API_TOKEN` and set `KAGGLE_USERNAME` to your account slug. Never commit credentials.
3. On Windows, double-click **launch-local.cmd**. It creates a virtual environment, installs dependencies, and opens the interface.

PowerShell alternative:

```powershell
.\start.ps1
# Or submit the notebook immediately:
.\start.ps1 -StartNotebook -Hours 3
```

Manual setup (Linux/macOS UI has not been tested):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py --open-browser
```

Open **http://127.0.0.1:7861**, click **Start notebook**, and wait for **Ready for chat**. The app creates/updates a private notebook named `deepseek-r1-32b-local-chat` in your account: do not use that slug for an unrelated notebook. An active run is not submitted twice.

**Stop Generation** cancels a response. **Stop notebook** requests session shutdown. During queueing/build/download, shutdown is deferred until the host API becomes ready; it is not an immediate Kaggle cancellation. Once ready, shutdown normally releases the VM within about 15 seconds. Closing the local app does not immediately stop Kaggle; use the shutdown button first. A session deadline and Kaggle execution timeout also apply.

## Chat and sharing

- Streaming responses, expandable reasoning, Markdown, code blocks, and LaTeX.
- New Chat, Clear conversation, Stop Generation, and Regenerate.
- Temperature 0.6, top_p 0.95, and requested max output tokens 8192 by default.
- Server token usage, elapsed time, generation time, and tokens/s when available.
- Context policies to retain recent turns or reject overflow while retaining all turns.

Input, history, reasoning, and output share the context window. An 8192-token output request is capped to the space remaining after input. Recent-turn mode may omit old pairs from a request; the latest prompt is preserved.

In **Share chat with other users**, click **Get shared chat login** for the temporary Kaggle Gradio URL and its username/password. Users have separate conversation state but share one inference slot and a queue of up to 32 requests, all against the owner's quota. The URL expires with the session. The shared chat password does not grant Kaggle access; backend shutdown additionally requires a separate admin token.

`python app.py --share` shares the local interface, including session controls, with authenticated trusted users. Prefer the Kaggle chat URL for ordinary guests. If Windows blocks Gradio's `frpc` executable, the local app still works through authenticated Kaggle live logs; use the Kaggle URL instead of sharing the local interface. Do not disable OS protection.

## Notebook

`notebook/deepseek.ipynb` is self-contained, with eight sections:

1. **Hardware check:** nvidia-smi, exactly two T4s, RAM, df -h, and writable scratch with at least 30 GiB free.
2. **Build llama.cpp:** clone/update upstream master, CUDA SM75 build, actual driver library discovery, and enumeration of both CUDA devices.
3. **Download model:** only Q4_K_M, outside `/kaggle/working`, with size and full SHA256 verification. Reuse a verified file within the same VM.
4. **Start server:** terminate the previous project server; all GPU layers, layer split 1:1, context 8192, flash attention, one sequence. CUDA OOM fallbacks reduce context to 6144, then 4096, before reducing offload. Never substitute another model.
5. **Server validation:** curl health, chat completion, reasoning detection, VRAM, and timings.
6. **Gradio UI:** authenticated sharing or inline display.
7. **Benchmark:** print summary; managed batch runs remain here until shutdown/deadline.
8. **Shutdown / cleanup:** stop the server and Gradio, retaining scratch files only within the current VM.

For interactive use, run cells 1-7; cell 8 explicitly shuts down. MODE A embeds the UI inline. Hosted Kaggle iframes need a share tunnel; MODE A also authenticates that transport and generates a random environment password if necessary. Browser cookie restrictions may require opening the URL in a separate tab. MODE B uses `share=True` and requires `GRADIO_PASSWORD` from an environment variable or Kaggle Secret. No password is hard-coded.

The local controller generates its managed notebook under `.runs/current`, passing only an RSA public key. The VM generates runtime credentials and encrypts the connection packet with RSA-OAEP before writing it to private logs. The local controller reads authenticated Kaggle live logs and decrypts with its local private key. llama-server itself binds only to `127.0.0.1:8080` inside Kaggle; remote chat goes through authenticated Gradio.

Regenerate the standalone example for your own account:

```bash
python -m model_host.notebook --owner YOUR_KAGGLE_USERNAME
```

## Verification and development

```bash
python -m pytest -q tests -p no:cacheprovider
# Requires your own running host:
python -m model_host.verify_remote
```

Tests cover input validation, context budgeting, notebook validity, authenticated Gradio streaming and cancellation with a simulated backend, local forwarding, OOM fallback order, encrypted per-run credentials, and launch modes. They do not replace GPU validation or evaluate model intelligence.

A T4 x2 trial used context 8192, all GPU layers, about 10481 MiB/GPU, roughly 71 prompt tok/s and 11 generation tok/s. These are measurements from one session, not performance guarantees. The real inference smoke test produced reasoning and the correct result for 17 x 19. Per-session evidence is local and excluded from Git.

`app.py` is the entry point. `model_host/controller.py` handles Kaggle and forwarding; `runtime.py` runs on Kaggle; `chat_ui.py` defines the shared UI; `notebook.py` generates notebooks. Set `HOST_DEBUG_QUEUE=1` only when diagnosing local queue state.

## Local data

`.state/` contains session credentials, private keys, and progress. `.runs/` contains generated managed notebooks. Logs, run evidence, model weights, environments, credentials, and internal agent instructions are ignored. Keep these local when distributing the project. The committed notebook has no outputs or session credentials.

Dependencies are pinned where applicable; llama.cpp follows upstream master, so future upstream changes may require compatibility fixes. This repository does not distribute model weights or llama.cpp. Their respective licenses apply to those downloads.

References: [Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [GGUF model](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF).
