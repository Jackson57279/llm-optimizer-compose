"""Vision/multimodal example with Miniforge."""

import asyncio
from pathlib import Path
from miniforge import Miniforge


async def main():
    """Run vision example."""

    print("Loading MiniMax M2.7 with vision...")

    model = await Miniforge.from_pretrained(
        "MiniMaxAI/MiniMax-M2.7",
        quantization="Q4_K_M",
    )

    print("Model loaded!\n")

    # Example image path (user needs to provide actual image)
    image_path = Path("example_image.jpg")

    if not image_path.exists():
        print(f"Note: Create an image at {image_path} to test vision")
        print("Skipping vision example...")
    else:
        # Vision chat
        response = await model.chat_vision(
            message="Describe what's in this image in detail.",
            image=image_path,
            max_tokens=512,
        )

        print(f"Vision response: {response}\n")

        # Another vision query
        response2 = await model.chat_vision(
            message="What objects do you see? List them.",
            image=image_path,
            max_tokens=256,
        )

        print(f"Objects: {response2}\n")

    await model.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
