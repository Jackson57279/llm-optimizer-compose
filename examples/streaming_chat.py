"""Streaming chat example."""

import asyncio
from miniforge import Miniforge


async def main():
    """Run streaming chat example."""

    print("Loading MiniMax M2.7...")

    model = await Miniforge.from_pretrained(
        "MiniMaxAI/MiniMax-M2.7",
        quantization="Q4_K_M",
    )

    print("Model loaded!\n")

    # Streaming response
    print("User: Write a short poem about AI\n")
    print("Assistant: ", end="", flush=True)

    stream = await model.chat(
        message="Write a short poem about AI",
        stream=True,
        max_tokens=256,
    )

    async for token in stream:
        print(token, end="", flush=True)

    print("\n")

    # Raw generation with streaming
    print("Prompt: Once upon a time\n")
    print("Response: ", end="", flush=True)

    gen_stream = await model.generate(
        prompt="Once upon a time",
        stream=True,
        max_tokens=100,
    )

    async for token in gen_stream:
        print(token, end="", flush=True)

    print("\n")

    await model.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
