"""Basic chat example with Miniforge."""

import asyncio
from miniforge import Miniforge
from miniforge.utils.config import M7Config


async def main():
    """Run basic chat example."""

    # Configure for GMKtech M7
    config = M7Config(
        quantization="Q4_K_M",
        cache_type_k="turbo3",
        cache_type_v="turbo3",
        n_threads=8,
        verbose=True,
    )

    print("Loading MiniMax M2.7...")

    # Initialize model
    model = await Miniforge.from_pretrained(
        model_id="MiniMaxAI/MiniMax-M2.7",
        config=config,
    )

    print("Model loaded!\n")

    # Simple chat
    response = await model.chat(
        message="Explain quantum computing in simple terms.",
        system_prompt="You are a helpful AI assistant.",
        max_tokens=512,
    )

    print(f"Assistant: {response}\n")

    # Chat with history
    history = [
        {"role": "user", "content": "Explain quantum computing in simple terms."},
        {"role": "assistant", "content": response},
    ]

    response2 = await model.chat(
        message="Can you give me a specific example?",
        history=history,
        system_prompt="You are a helpful AI assistant.",
    )

    print(f"Assistant: {response2}\n")

    # Print memory stats
    stats = model.get_memory_stats()
    print(f"Memory usage: {stats}")

    # Cleanup
    await model.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
