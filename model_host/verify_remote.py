"""Verify the real local → Gradio → llama.cpp → GPU inference path."""
import asyncio
from datetime import datetime, timezone
import json
from .controller import Controller
from .notebook import ROOT


async def verify():
    controller = Controller(monitor=False)
    controller.poll()
    if not controller.host_auth:
        raise RuntimeError("Host is not ready: " + controller.report.get("stage", "unknown"))
    folder = ROOT / "validation"
    folder.mkdir(exist_ok=True)
    client = controller.client()
    try:
        before = client.predict(api_name="/host_status")
    finally:
        client.close()
    if not before.get("reasoning_verified"):
        raise RuntimeError("Notebook has not confirmed reasoning yet")
    chunks = 0
    final = None
    async for result in controller.stream([
        {"role": "user", "content": "Tính 17 nhân 19. Trả lời kết quả ngắn gọn bằng tiếng Việt."}
    ], 0.6, 0.95, 512, "Keep recent turns"):
        chunks += 1
        final = result
    if not final or chunks < 2:
        raise RuntimeError("No streaming response received")
    answer = final["messages"][-1]
    if "323" not in answer.get("content", ""):
        raise RuntimeError("Arithmetic validation did not produce the expected final answer: " + answer.get("content", ""))
    if not answer.get("reasoning_content") and "<think>" not in answer.get("content", ""):
        raise RuntimeError("No reasoning trace in the real inference response")
    client = controller.client()
    try:
        status = client.predict(api_name="/host_status")
    finally:
        client.close()
    status["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    status["local_stream_verified"] = True
    status["stream_chunks"] = chunks
    (folder / "summary.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    (folder / "remote-chat.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stream_chunks": chunks, "answer": answer["content"],
        "usage": final.get("usage"), "timings": final.get("timings"),
        "summary": {k: status.get(k) for k in ("Model", "Quantization", "Model size", "Context", "GPU 0 VRAM",
        "GPU 1 VRAM", "Prompt processing tok/s", "Generation tok/s", "Backend URL", "Gradio URL")}},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(verify())
