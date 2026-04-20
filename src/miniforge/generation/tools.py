"""Tool calling support for MiniMax models."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call from the model."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""

    tool_call_id: str
    content: str
    is_error: bool = False


class Tool:
    """Definition of a tool the model can use."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Optional[Callable] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolExecutor:
    """
    Execute tool calls from model output.

    Supports async handlers and error handling.
    """

    def __init__(self, tools: Optional[List[Tool]] = None):
        self.tools = {t.name: t for t in (tools or [])}
        self._handlers: Dict[str, Callable] = {}

    def register(self, tool: Tool, handler: Optional[Callable] = None) -> None:
        """Register a tool with optional handler."""
        self.tools[tool.name] = tool
        if handler:
            self._handlers[tool.name] = handler
        elif tool.handler:
            self._handlers[tool.name] = tool.handler

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a handler for an existing tool."""
        self._handlers[name] = handler

    async def execute(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool results
        """
        results = []
        for call in tool_calls:
            result = await self.execute_single(call)
            results.append(result)
        return results

    async def execute_single(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        name = tool_call.name

        if name not in self.tools:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: Tool '{name}' not found",
                is_error=True,
            )

        if name not in self._handlers:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: No handler registered for tool '{name}'",
                is_error=True,
            )

        handler = self._handlers[name]

        try:
            # Call handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**tool_call.arguments)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: handler(**tool_call.arguments))

            return ToolResult(
                tool_call_id=tool_call.id,
                content=str(result),
                is_error=False,
            )
        except Exception as e:
            logger.error(f"Tool execution error for {name}: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error executing {name}: {str(e)}",
                is_error=True,
            )

    def format_tools_for_prompt(self, tools: List[Tool]) -> str:
        """Format tools for inclusion in system prompt."""
        tool_descriptions = []
        for tool in tools:
            tool_descriptions.append(
                f"Tool: {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Parameters: {json.dumps(tool.parameters, indent=2)}"
            )

        return (
            "You have access to the following tools:\n\n"
            + "\n\n".join(tool_descriptions)
            + "\n\nTo use a tool, respond with:\n"
            "<tool>\n"
            '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
            "</tool>"
        )

    def parse_tool_calls(self, text: str) -> List[ToolCall]:
        """
        Parse tool calls from model output.

        Supports multiple formats:
        - XML tags: <tool>...</tool>
        - JSON blocks
        - OpenAI format
        """
        tool_calls = []

        # Try XML format
        import re

        xml_pattern = r"<tool>(.*?)</tool>"
        for match in re.finditer(xml_pattern, text, re.DOTALL):
            try:
                data = json.loads(match.group(1).strip())
                tool_calls.append(
                    ToolCall(
                        id=data.get("id", f"call_{len(tool_calls)}"),
                        name=data.get("name", data.get("function", {}).get("name")),
                        arguments=data.get(
                            "arguments", data.get("function", {}).get("arguments", {})
                        ),
                    )
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match.group(1)}")

        # Try JSON format if no XML found
        if not tool_calls:
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "name" in data:
                    tool_calls.append(
                        ToolCall(
                            id="call_0",
                            name=data["name"],
                            arguments=data.get("arguments", {}),
                        )
                    )
            except json.JSONDecodeError:
                pass

        return tool_calls


# Pre-built common tools


def create_calculator_tool() -> Tool:
    """Create a calculator tool."""

    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            # Safe eval with limited scope
            allowed_names = {
                "abs": abs,
                "max": max,
                "min": min,
                "pow": pow,
                "round": round,
                "sum": sum,
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return Tool(
        name="calculator",
        description="Evaluate mathematical expressions",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
        handler=calculator,
    )


def create_web_search_tool(search_func: Optional[Callable] = None) -> Tool:
    """Create a web search tool."""
    return Tool(
        name="web_search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
        handler=search_func,
    )
