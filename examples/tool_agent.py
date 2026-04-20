"""Tool calling example with Miniforge."""

import asyncio
from miniforge import Miniforge
from miniforge.generation.tools import (
    Tool,
    ToolExecutor,
    create_calculator_tool,
)


async def get_weather(location: str) -> str:
    """Mock weather function."""
    # In real usage, call actual weather API
    return f"Weather in {location}: Sunny, 72F, light breeze"


async def search_web(query: str) -> str:
    """Mock web search function."""
    return f"Search results for '{query}': [Mock results]"


async def main():
    """Run tool calling example."""

    print("Loading MiniMax M2.7 with tools...")

    model = await Miniforge.from_pretrained(
        "MiniMaxAI/MiniMax-M2.7",
        quantization="Q4_K_M",
    )

    print("Model loaded!\n")

    # Create tools
    calculator = create_calculator_tool()

    weather_tool = Tool(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or location name",
                }
            },
            "required": ["location"],
        },
        handler=get_weather,
    )

    search_tool = Tool(
        name="search_web",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                }
            },
            "required": ["query"],
        },
        handler=search_web,
    )

    tools = [calculator, weather_tool, search_tool]

    # Chat with tools
    response = await model.chat(
        message="What's the weather like in San Francisco? Also, calculate 15 * 23.",
        tools=tools,
        max_tokens=256,
    )

    print(f"Response: {response}\n")

    # Another tool use example
    response2 = await model.chat(
        message="Search for recent developments in quantum computing",
        tools=tools,
        max_tokens=256,
    )

    print(f"Response: {response2}\n")

    await model.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
