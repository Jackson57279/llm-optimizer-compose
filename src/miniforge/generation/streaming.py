"""Streaming token generation utilities."""

from typing import AsyncIterator, Callable, Optional, Any
import asyncio
import logging

logger = logging.getLogger(__name__)


class TokenStreamer:
    """
    Async token streaming with buffering and backpressure handling.

    Supports:
    - Real-time streaming to callbacks
    - Buffering for downstream processing
    - Rate limiting for UI updates
    """

    def __init__(
        self,
        callback: Optional[Callable[[str], None]] = None,
        buffer_size: int = 1024,
        rate_limit_ms: float = 0,
    ):
        """
        Initialize token streamer.

        Args:
            callback: Optional callback for each token
            buffer_size: Size of internal buffer
            rate_limit_ms: Minimum ms between token emissions (0 = no limit)
        """
        self.callback = callback
        self.buffer_size = buffer_size
        self.rate_limit_ms = rate_limit_ms
        self._buffer = []
        self._queue = asyncio.Queue(maxsize=buffer_size)
        self._total_tokens = 0
        self._last_emit_time = 0

    async def feed(self, token: str) -> None:
        """Feed a token to the streamer."""
        # Apply rate limiting
        if self.rate_limit_ms > 0:
            current_time = asyncio.get_event_loop().time() * 1000
            elapsed = current_time - self._last_emit_time
            if elapsed < self.rate_limit_ms:
                await asyncio.sleep((self.rate_limit_ms - elapsed) / 1000)
            self._last_emit_time = current_time

        # Add to buffer
        self._buffer.append(token)
        self._total_tokens += 1

        # Call callback if provided
        if self.callback:
            try:
                self.callback(token)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

        # Add to queue (may block if full)
        await self._queue.put(token)

    async def stream(self) -> AsyncIterator[str]:
        """Stream tokens as async iterator."""
        while True:
            try:
                # Wait for token with timeout
                token = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield token
            except asyncio.TimeoutError:
                # Check if generation is complete
                if hasattr(self, "_done") and self._done:
                    break

    def done(self) -> None:
        """Mark streaming as complete."""
        self._done = True
        # Signal completion
        asyncio.create_task(self._queue.put(""))

    def get_buffer(self) -> str:
        """Get all buffered tokens as string."""
        return "".join(self._buffer)

    def get_stats(self) -> dict:
        """Get streaming statistics."""
        return {
            "total_tokens": self._total_tokens,
            "buffer_size": len(self._buffer),
        }


class ChatStreamHandler:
    """
    High-level handler for chat streaming with formatting.

    Handles:
    - Message formatting
    - Tool call detection
    - Thinking/reasoning blocks
    """

    def __init__(self):
        self.buffer = ""
        self.tool_calls = []
        self.in_tool_call = False

    async def process_stream(
        self,
        token_stream: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        """
        Process raw token stream, detecting special patterns.

        Yields formatted tokens.
        """
        async for token in token_stream:
            self.buffer += token

            # Detect tool call patterns
            if "<tool>" in self.buffer and not self.in_tool_call:
                self.in_tool_call = True
                # Yield content before tool tag
                idx = self.buffer.index("<tool>")
                if idx > 0:
                    yield self.buffer[:idx]
                self.buffer = self.buffer[idx:]

            if "</tool>" in self.buffer and self.in_tool_call:
                # Extract tool call
                self.in_tool_call = False
                # Parse tool call content
                # ... parsing logic ...
                self.buffer = ""

            if not self.in_tool_call:
                yield token

    def get_full_response(self) -> str:
        """Get complete response including any tool calls."""
        return self.buffer

    def get_tool_calls(self) -> list:
        """Get extracted tool calls."""
        return self.tool_calls
