"""Runs only inside the Kaggle notebook. No agent or model tools."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import secrets
import base64

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
REPO = "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF"
FILENAME = "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
WORK = Path("/kaggle/working/deepseek-host")
BASE_URL = "http://127.0.0.1:8080"
STOP = threading.Event()
REPORT = {}
CONFIG = {}
PROCESS = None


def run(args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, text=True, **kwargs)


def save():
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "summary.json").write_text(json.dumps(REPORT, indent=2))


def notify(stage, **details):
    REPORT.update(details, stage=stage, run_id=CONFIG.get("run_id"))
    save()
    print("HOST_PROGRESS:" + json.dumps(REPORT), flush=True)
    if STOP.is_set() and stage != "Stopped":
        raise RuntimeError("Session stopped from the local interface")


def hardware():
    run(["nvidia-smi"])
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"
    ], text=True)
    rows = [r.split(",") for r in devices.strip().splitlines()]
    if len(rows) != 2 or any("T4" not in r[1] or int(r[2]) < 14500 for r in rows):
        raise RuntimeError("Required exactly two NVIDIA T4 GPUs with nominal 16 GB each: " + devices)
    print(Path("/proc/meminfo").read_text().split("SwapCached")[0])
    run(["free", "-h"])
    run(["df", "-h"])
    WORK.mkdir(parents=True, exist_ok=True)
    candidates = ["/kaggle/temp", "/tmp", "/var/tmp"]
    candidates += [line.split()[1] for line in Path("/proc/mounts").read_text().splitlines()
                   if len(line.split()) > 2 and line.split()[2] in {"ext4", "xfs", "overlay"}]
    observations = []
    for candidate in dict.fromkeys(candidates):
        p = Path(candidate).resolve() / "deepseek-scratch"
        if p.is_relative_to(Path("/kaggle/working")) or str(p).startswith("/kaggle/input"):
            continue
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".writable"
            probe.write_text("ok")
            probe.unlink()
            free = shutil.disk_usage(p).free
            existing = p / "models" / FILENAME
            reusable = existing.exists() and existing.with_suffix(".verified.json").exists()
            observations.append(f"{p}: {free / 2**30:.1f} GiB free")
            if free >= 30 * 2**30 or (reusable and free >= 3 * 2**30):
                CONFIG["scratch"] = str(p)
                REPORT.update(hardware=devices, scratch=str(p), free_bytes=free)
                save()
                print("Selected writable scratch:", observations[-1])
                return
        except OSError as exc:
            observations.append(f"{p}: {exc}")
    raise RuntimeError("No writable scratch with at least 30 GiB free. No model download started. "
                       + "; ".join(observations))


def build():
    run([sys.executable, "-m", "pip", "install", "gradio==6.30.0", "gradio_client==2.7.2",
         "huggingface_hub>=0.30,<3", "requests>=2.32,<3", "psutil>=5.9", "cryptography>=44,<50"])
    bootstrap()
    notify("Building llama.cpp with CUDA")
    repo = Path(CONFIG["scratch"]) / "llama.cpp"
    if not (repo / ".git").exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", repo])
    else:
        shutdown_server()
        run(["git", "-C", repo, "fetch", "--depth", "1", "origin", "master"])
        run(["git", "-C", repo, "merge", "--ff-only", "FETCH_HEAD"])
    args = ["cmake", "-S", repo, "-B", repo / "build", "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"]
    # Kaggle mounts the real driver outside the toolkit's library search paths.
    import ctypes
    ctypes.CDLL("libcuda.so.1")
    drivers = [Path(line.split()[-1]) for line in Path("/proc/self/maps").read_text().splitlines()
               if "/libcuda.so" in line and "stubs" not in line]
    if not drivers:
        raise RuntimeError("CUDA driver loaded but its filesystem path could not be resolved")
    driver = drivers[0].resolve()
    print("Actual CUDA driver library:", driver, flush=True)
    args += [f"-DCUDA_cuda_driver_LIBRARY={driver}", f"-DCMAKE_LIBRARY_PATH={driver.parent}"]
    run(args)
    jobs = max(1, min(4, os.cpu_count() or 2))
    try:
        run(["cmake", "--build", repo / "build", "--target", "llama-server", "-j", jobs])
    except subprocess.CalledProcessError:
        print("Retrying build with one compiler process to reduce host RAM pressure.")
        run(["cmake", "--build", repo / "build", "--target", "llama-server", "-j", "1"])
    CONFIG["binary"] = str(repo / "build/bin/llama-server")
    devices = subprocess.check_output([CONFIG["binary"], "--list-devices"],
                                      text=True, stderr=subprocess.STDOUT)
    print(devices)
    if not all(re.search(rf"CUDA:?{i}\b", devices) for i in (0, 1)):
        raise RuntimeError("llama-server did not enumerate both CUDA:0 and CUDA:1")
    REPORT["llama_commit"] = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    save()


def bootstrap():
    # In managed batch runs credentials are generated in the VM, never embedded in source.
    if CONFIG.get("public_key"):
        os.environ.setdefault("GRADIO_USER", "model-user")
        if not os.environ.get("GRADIO_PASSWORD"):
            try:
                from kaggle_secrets import UserSecretsClient
                os.environ["GRADIO_PASSWORD"] = UserSecretsClient().get_secret("GRADIO_PASSWORD")
            except Exception:
                os.environ["GRADIO_PASSWORD"] = secrets.token_urlsafe(24)
        CONFIG["admin_token"] = secrets.token_urlsafe(32)


def publish_access(url):
    if not CONFIG.get("public_key"):
        return
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    public = serialization.load_pem_public_key(CONFIG["public_key"].encode())
    packet = json.dumps({"url": url, "username": os.environ["GRADIO_USER"],
                         "password": os.environ["GRADIO_PASSWORD"], "admin_token": CONFIG["admin_token"],
                         "run_id": CONFIG["run_id"]}).encode()
    encrypted = public.encrypt(packet, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                                   algorithm=hashes.SHA256(), label=None))
    print("HOST_ACCESS:" + base64.b64encode(encrypted).decode(), flush=True)


def download():
    from huggingface_hub import HfApi, hf_hub_download
    notify("Checking exact GGUF file and scratch capacity")
    folder = Path(CONFIG["scratch"]) / "models"
    folder.mkdir(exist_ok=True)
    model = folder / FILENAME
    marker = model.with_suffix(".verified.json")
    info = HfApi().model_info(REPO, files_metadata=True)
    file = next(s for s in info.siblings if s.rfilename == FILENAME)
    expected_size = file.size
    expected_hash = getattr(file.lfs, "sha256", None)
    if not expected_size or not expected_hash:
        raise RuntimeError("Hugging Face did not return file size and SHA256; refusing unverifiable download")
    verified = json.loads(marker.read_text()) if marker.exists() else {}
    valid = (model.exists() and model.stat().st_size == expected_size
             and verified.get("sha256") == expected_hash
             and verified.get("mtime_ns") == model.stat().st_mtime_ns)
    if not valid:
        free = shutil.disk_usage(folder).free
        if free < max(30 * 2**30, expected_size + 2 * 2**30):
            raise RuntimeError(f"Scratch only has {free/2**30:.1f} GiB free; need at least 30 GiB. Download stopped.")
        notify("Downloading Q4_K_M only", model_bytes=expected_size)
        model = Path(hf_hub_download(REPO, FILENAME, revision=info.sha, local_dir=folder))
        if model.stat().st_size != expected_size:
            raise RuntimeError("Incomplete GGUF download")
        digest = hashlib.sha256()
        with model.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise RuntimeError("GGUF SHA256 mismatch; refusing to load")
        marker.write_text(json.dumps({"sha256": expected_hash, "mtime_ns": model.stat().st_mtime_ns}))
    else:
        print("Complete, previously SHA256-verified file exists; skipping download.")
    CONFIG["model"] = str(model)
    REPORT.update(model=MODEL, quantization="Q4_K_M", model_bytes=expected_size, model_revision=info.sha)
    save()


def shutdown_server():
    import psutil
    binary = CONFIG.get("binary")
    if not binary:
        return
    targets = []
    for proc in psutil.process_iter(["exe", "cmdline"]):
        try:
            if proc.info["exe"] == binary or (proc.info["cmdline"] and proc.info["cmdline"][0] == binary):
                proc.terminate()
                targets.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(targets, timeout=10)
    for proc in alive:
        proc.kill()
    psutil.wait_procs(alive, timeout=10)


def start():
    import requests
    global PROCESS
    shutdown_server()
    help_text = subprocess.check_output([CONFIG["binary"], "--help"], text=True, stderr=subprocess.STDOUT)
    flash = "on"
    attempts = [(8192, "all"), (6144, "all"), (4096, "all")]
    attempts += [(4096, str(n)) for n in (60, 56, 48, 40, 32, 16, 8, 4, 2)]
    for ctx, layers in attempts:
        while True:
            notify("Loading model", context=ctx, gpu_layers=layers, flash_attention=flash)
            args = [CONFIG["binary"], "--model", CONFIG["model"], "--host", "127.0.0.1", "--port", "8080",
                    "--device", "CUDA0,CUDA1", "--n-gpu-layers", layers, "--split-mode", "layer",
                    "--tensor-split", "1,1", "--ctx-size", str(ctx), "--parallel", "1",
                    "--flash-attn", flash, "--temp", "0.6", "--top-p", "0.95", "--top-k", "0",
                    "--min-p", "0", "--repeat-penalty", "1", "--jinja", "--reasoning-format", "deepseek",
                    "--batch-size", "512", "--ubatch-size", "128"]
            if "--fit" in help_text:
                args += ["--fit", "off"]
            log_path = WORK / "llama-server.log"
            with log_path.open("w") as log:
                PROCESS = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            (WORK / "server.pid").write_text(str(PROCESS.pid))
            deadline = time.monotonic() + 900
            while PROCESS.poll() is None and time.monotonic() < deadline:
                if STOP.is_set():
                    shutdown_server()
                    raise RuntimeError("Session stopped during model loading")
                try:
                    response = requests.get(BASE_URL + "/health", timeout=2)
                    if response.status_code == 200:
                        CONFIG["context"] = ctx
                        notify("Backend ready", backend_url=BASE_URL, server_command=args)
                        return
                except requests.RequestException:
                    pass
                time.sleep(2)
            shutdown_server()
            logs = log_path.read_text(errors="replace")
            print(logs[-16000:])
            if flash == "on" and re.search(r"flash.*(?:not supported|unsupported|requires)|(?:not supported|unsupported).*flash", logs, re.I):
                flash = "off"
                print("Flash attention unsupported; retrying same memory configuration without it.")
                continue
            if re.search(r"out of memory|cudaMalloc.*failed|CUDA error.*memory|failed to allocate.*(?:CUDA|device)", logs, re.I):
                print("CUDA OOM: advancing context/offload fallback in the prescribed order.")
                break
            raise RuntimeError("llama-server failed for a non-OOM reason. See llama-server.log: " + logs[-3000:])
    raise RuntimeError("Insufficient VRAM after context and GPU offload fallbacks; model was not replaced.")


def vram():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                                   "--format=csv,noheader,nounits"], text=True)
    REPORT["gpu_vram"] = raw.strip().splitlines()
    return raw


def completion(prompt, tokens=512):
    import requests
    response = requests.post(BASE_URL + "/v1/chat/completions", json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens, "temperature": 0.6, "top_p": 0.95, "cache_prompt": False
    }, timeout=600)
    response.raise_for_status()
    return response.json()


def validate():
    run(["curl", "--fail", "--silent", "--show-error", BASE_URL + "/health"])
    result = completion("Find the smallest positive integer n for which n leaves remainder 2 when divided by 3, and remainder 3 when divided by 5. Verify the result.", 1024)
    message = result["choices"][0]["message"]
    print(json.dumps(result, indent=2, ensure_ascii=False))
    reasoning = message.get("reasoning_content") or ("<think>" in message.get("content", ""))
    if not reasoning:
        raise RuntimeError("Validation returned no reasoning trace; inspect GGUF chat template and server log.")
    REPORT.update(reasoning_verified=True, validation_usage=result.get("usage"), timings=result.get("timings", {}))
    print(vram())
    save()


def benchmark():
    result = completion("Explain briefly how to check whether 97 is prime, then give the conclusion.", 256)
    REPORT["timings"] = result.get("timings", {})
    print(vram())
    notify("Ready for chat", **summary())
    print(json.dumps(summary(), indent=2, ensure_ascii=False))


def summary():
    timing = REPORT.get("timings", {})
    gpu = REPORT.get("gpu_vram", [])
    return {"Model": MODEL, "Quantization": "Q4_K_M", "Model size": REPORT.get("model_bytes"),
            "Context": CONFIG.get("context"), "GPU 0 VRAM": gpu[0] if gpu else None,
            "GPU 1 VRAM": gpu[1] if len(gpu) > 1 else None,
            "Prompt processing tok/s": timing.get("prompt_per_second"),
            "Generation tok/s": timing.get("predicted_per_second"),
            "Backend URL": BASE_URL, "Gradio URL": REPORT.get("gradio_url")}


def shutdown():
    shutdown_server()
    notify("Stopped", stopped=True)
    demo = CONFIG.get("demo")
    if demo:
        demo.close()


def keep_alive():
    """Batch Kaggle runs must stay inside a cell, otherwise the GPU VM exits."""
    deadline = time.monotonic() + float(CONFIG.get("session_hours", 3)) * 3600
    print("Session active. Use Stop notebook in the local interface to release GPUs.", flush=True)
    try:
        while not STOP.wait(15):
            if time.monotonic() >= deadline:
                break
            if PROCESS and PROCESS.poll() is not None:
                raise RuntimeError("llama-server exited unexpectedly")
            notify("Ready for chat")
    finally:
        shutdown()
