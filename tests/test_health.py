"""Quick health check test."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from app.main import app


async def main():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://local",
    ) as client:
        resp = await client.get("/health")
        print("health:", resp.status_code, resp.json())
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


if __name__ == "__main__":
    asyncio.run(main())
