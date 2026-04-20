"""Example OpenAI-compatible API server."""

import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from miniforge import Miniforge
from miniforge.utils.config import M7Config


# Simple async server example using aiohttp
# For production, use FastAPI or similar


async def main():
    """Run example API server."""

    print("Miniforge API Server Example")
    print("=" * 40)

    config = M7Config.from_yaml("configs/m7-optimized.yaml")

    print("Loading model...")
    model = await Miniforge.from_pretrained(
        "MiniMaxAI/MiniMax-M2.7",
        config=config,
    )

    print(f"Model loaded!")
    print(f"Memory stats: {model.get_memory_stats()}")

    # Simple chat test
    print("\nTest chat:")
    response = await model.chat("Hello! What can you help me with?")
    print(f"Response: {response}")

    # Keep running (in real server, this would be the API loop)
    print("\nServer would run here...")
    print("Press Ctrl+C to exit")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

    await model.cleanup()
    print("Server stopped.")


if __name__ == "__main__":
    asyncio.run(main())
