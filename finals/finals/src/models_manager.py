import json
from base64 import b64encode
from typing import Any

import httpx
import websockets


class ModelsManager:
    def __init__(self, local_ip: str):
        self.local_ip = local_ip
        print("initializing participant finals server manager")
        self.client = httpx.AsyncClient()

    async def exit(self):
        await self.client.aclose()

    async def async_post(self, endpoint: str, json: dict | None = None):
        return await self.client.post(endpoint, json=json, timeout=None)

    async def send_result(
        self, websocket: websockets.ClientConnection, data: dict[str, Any]
    ):
        return await websocket.send(json.dumps(data))

    async def run_asr(self, audio_b64: str) -> str:
        print("Running ASR")
        results = await self.async_post(
            f"http://{self.local_ip}:5001/asr",
            json={"instances": [{"b64": audio_b64}]},
        )
        return results.json()["predictions"][0]

    async def run_cv(self, image_b64: str) -> list[int]:
        print("Running CV")
        results = await self.async_post(
            f"http://{self.local_ip}:5002/cv",
            json={"instances": [{"b64": image_b64}]},
        )
        return results.json()["predictions"][0]

    async def run_ocr(self, image_b64: str) -> str:
        print("Running OCR")
        results = await self.async_post(
            f"http://{self.local_ip}:5003/ocr",
            json={"instances": [{"b64": image_b64}]},
        )
        return results.json()["predictions"][0]

    async def run_rl(self, observation: dict[str, int | list[int]]) -> int:
        print("Running RL")
        results = await self.async_post(
            f"http://{self.local_ip}:5004/rl",
            json={"instances": [{"observation": observation}]},
        )
        return results.json()["predictions"][0]["action"]
    
    async def run_surprise(self, slices: list[str]) -> list[int]:
        print("Running Surprise")
        results = await self.async_port(
            f"http://{self.local_ip}:5005/surprise",
            json={"instances": [{"slices": slices}]}
        )
        return results.json()["predictions"][0]