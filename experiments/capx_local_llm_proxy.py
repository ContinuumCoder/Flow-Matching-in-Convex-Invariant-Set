

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "local/deepseek-r1-distill-qwen-32b"
    messages: list[ChatMessage]
    temperature: float | None = 0.2
    max_tokens: int | None = 1536
    stream: bool = False


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
        return "\n".join(pieces)
    return str(content)


def _build_prompt(tokenizer: AutoTokenizer, messages: list[ChatMessage]) -> str:
    chat = [{"role": m.role, "content": _content_to_text(m.content)} for m in messages]
    try:
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except Exception:
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in chat) + "\n\nassistant:"


def create_app(model_path: str, max_model_len: int | None) -> FastAPI:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if max_model_len is not None:
        kwargs["max_position_embeddings"] = max_model_len
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()

    app = FastAPI(title="Local HF Chat Proxy")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat/completions")
    def chat(req: ChatCompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise ValueError("Streaming is not implemented in this lightweight proxy.")
        prompt = _build_prompt(tokenizer, req.messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        max_new_tokens = int(req.max_tokens or 1536)
        do_sample = (req.temperature or 0.0) > 1e-6
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=float(req.temperature or 1.0) if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {
            "id": f"local-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("CAPX_LOCAL_MODEL_PATH", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--max-model-len", type=int, default=None)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model_path, args.max_model_len), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
