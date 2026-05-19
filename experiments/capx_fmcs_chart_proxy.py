

from __future__ import annotations

import argparse
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


STACK_CODE = """
import numpy as np

_, _, green_ext = get_object_pose("green cube", return_bbox_extent=True)
_, _, red_ext = get_object_pose("red cube", return_bbox_extent=True)

pick_pos, pick_quat = sample_grasp_pose("red cube")
goto_pose(pick_pos, pick_quat, z_approach=0.1)
close_gripper()

post_pick_pos = pick_pos.copy()
post_pick_pos[2] += 0.2
goto_pose(post_pick_pos, pick_quat)

green_pos, _, _ = get_object_pose("green cube", return_bbox_extent=False)
place_pos = green_pos.copy()
place_pos[2] = green_pos[2] + green_ext[2] / 2 + red_ext[2] / 2

goto_pose(place_pos, pick_quat, z_approach=0.1)
open_gripper()

post_place_pos = place_pos.copy()
post_place_pos[2] += 0.1
goto_pose(post_place_pos, pick_quat)
"""


RESTACK_CODE = """
import numpy as np

_, _, green_ext = get_object_pose("green cube", return_bbox_extent=True)
_, _, red_ext = get_object_pose("red cube", return_bbox_extent=True)

green_pick_pos, green_pick_quat = sample_grasp_pose("green cube")
goto_pose(green_pick_pos, green_pick_quat, z_approach=0.1)
close_gripper()
green_clear_pos = green_pick_pos.copy()
green_clear_pos[0] -= 0.15
green_clear_pos[2] += 0.1
goto_pose(green_clear_pos, green_pick_quat)
open_gripper()

red_pick_pos, red_pick_quat = sample_grasp_pose("red cube")
goto_pose(red_pick_pos, red_pick_quat, z_approach=0.1)
close_gripper()
red_lift_pos = red_pick_pos.copy()
red_lift_pos[0] -= 0.15
red_lift_pos[2] += 0.3
goto_pose(red_lift_pos, red_pick_quat)

green_pos, _, _ = get_object_pose("green cube", return_bbox_extent=False)
place_pos = green_pos.copy()
place_pos[2] = green_pos[2] + green_ext[2] + red_ext[2]

goto_pose(place_pos, red_pick_quat, z_approach=0.1)
open_gripper()

post_place_pos = place_pos.copy()
post_place_pos[2] += 0.1
goto_pose(post_place_pos, red_pick_quat)
"""


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "fmcs-chart-proxy"
    messages: list[ChatMessage]
    temperature: float | None = 0.0
    max_tokens: int | None = 2048
    stream: bool = False


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _select_code(messages: list[ChatMessage]) -> str:
    prompt = "\n".join(_content_to_text(message.content) for message in messages).lower()
    if "restack" in prompt or "re-stack" in prompt or "unstack" in prompt:
        return RESTACK_CODE
    return STACK_CODE


def create_app() -> FastAPI:
    app = FastAPI(title="FMCS Chart Proxy")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat/completions")
    def chat(req: ChatCompletionRequest) -> dict[str, Any]:
        code = RESTACK_CODE if "restack" in req.model.lower() else _select_code(req.messages)
        return {
            "id": f"fmcs-chart-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": code},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8112)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
