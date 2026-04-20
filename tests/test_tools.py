"""Test tool calling functionality."""

import pytest
import asyncio

from miniforge.generation.tools import (
    Tool,
    ToolCall,
    ToolResult,
    ToolExecutor,
    create_calculator_tool,
)


def test_tool_creation():
    """Test tool creation."""
    tool = Tool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
    )

    assert tool.name == "test_tool"
    assert tool.description == "A test tool"


def test_tool_to_dict():
    """Test tool serialization."""
    tool = Tool(
        name="get_weather",
        description="Get weather",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    )

    data = tool.to_dict()
    assert data["type"] == "function"
    assert data["function"]["name"] == "get_weather"


def test_tool_executor_registration():
    """Test tool registration."""
    executor = ToolExecutor()

    tool = Tool(
        name="test",
        description="Test",
        parameters={},
        handler=lambda: "result",
    )

    executor.register(tool)
    assert "test" in executor.tools


@pytest.mark.asyncio
async def test_tool_execution():
    """Test tool execution."""
    executor = ToolExecutor()

    async def mock_handler(x: int) -> str:
        return f"Result: {x * 2}"

    tool = Tool(
        name="double",
        description="Double a number",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        },
        handler=mock_handler,
    )

    executor.register(tool, mock_handler)

    call = ToolCall(id="call_1", name="double", arguments={"x": 5})
    result = await executor.execute_single(call)

    assert not result.is_error
    assert "10" in result.content


@pytest.mark.asyncio
async def test_tool_execution_error():
    """Test tool execution with error."""
    executor = ToolExecutor()

    def bad_handler():
        raise ValueError("Test error")

    tool = Tool(
        name="bad_tool",
        description="Always fails",
        parameters={},
        handler=bad_handler,
    )

    executor.register(tool)

    call = ToolCall(id="call_1", name="bad_tool", arguments={})
    result = await executor.execute_single(call)

    assert result.is_error
    assert "Error" in result.content


@pytest.mark.asyncio
async def test_tool_not_found():
    """Test tool not found error."""
    executor = ToolExecutor()

    call = ToolCall(id="call_1", name="missing", arguments={})
    result = await executor.execute_single(call)

    assert result.is_error
    assert "not found" in result.content


def test_parse_tool_calls_xml():
    """Test parsing XML tool calls."""
    executor = ToolExecutor()

    text = """
    Some text
    <tool>
    {"name": "search", "arguments": {"query": "python"}}
    </tool>
    More text
    """

    calls = executor.parse_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments["query"] == "python"


def test_parse_tool_calls_json():
    """Test parsing JSON tool calls."""
    executor = ToolExecutor()

    text = '{"name": "calc", "arguments": {"x": 1, "y": 2}}'

    calls = executor.parse_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].name == "calc"


def test_calculator_tool():
    """Test built-in calculator tool."""
    tool = create_calculator_tool()

    # Test handler
    result = tool.handler("2 + 2")
    assert "4" in result

    result2 = tool.handler("10 * 5")
    assert "50" in result2
