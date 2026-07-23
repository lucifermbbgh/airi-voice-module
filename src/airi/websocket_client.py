"""
AIRI WebSocket client module.

Connects to AIRI's plugin protocol WebSocket server and communicates
using the Eventa-based event system.

Phase 1: Basic connection, authentication, heartbeat, and event listening.
Future phases: Send input:text:voice events, receive TTS responses.
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any

import websockets

from src.logger import get_logger

logger = get_logger(__name__)


class AIRIEventType(str, Enum):
    """AIRI WebSocket event types relevant to voice module."""
    INPUT_TEXT = "input:text"
    INPUT_TEXT_VOICE = "input:text:voice"
    INPUT_VOICE = "input:voice"
    OUTPUT_CHAT_MESSAGE = "output:gen-ai:chat:message"
    OUTPUT_CHAT_COMPLETE = "output:gen-ai:chat:complete"
    TRANSPORT_HEARTBEAT = "transport:connection:heartbeat"
    ERROR = "error"


class AIRIClient:
    """AIRI plugin protocol WebSocket client.

    Attributes:
        host: AIRI server hostname.
        port: AIRI WebSocket port.
        token: Authentication token.
        reconnect_interval: Seconds between reconnect attempts.
        max_attempts: Maximum reconnect attempts (0 = unlimited).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 10443,
        token: str = "",
        reconnect_interval: int = 5,
        max_attempts: int = 0,
    ):
        """Initialize AIRI WebSocket client.

        Args:
            host: AIRI server hostname.
            port: WebSocket port.
            token: Authentication token.
            reconnect_interval: Seconds between reconnects.
            max_attempts: Max reconnect attempts (0 = unlimited).
        """
        self.host = host
        self.port = port
        self.token = token
        self.reconnect_interval = reconnect_interval
        self.max_attempts = max_attempts

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._running = False
        self._connected = False
        self._event_handlers: dict[str, list[callable]] = {}

    @property
    def url(self) -> str:
        """Get WebSocket URL."""
        return f"ws://{self.host}:{self.port}"

    @property
    def is_connected(self) -> bool:
        """Check if connected to AIRI."""
        return self._connected

    def on(self, event_type: str, handler: callable) -> None:
        """Register an event handler.

        Args:
            event_type: Event type string (e.g., "input:text:voice").
            handler: Async callback function(event_data: dict).
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def _dispatch_event(self, event_type: str, data: dict) -> None:
        """Dispatch an event to registered handlers.

        Args:
            event_type: Event type string.
            data: Event payload.
        """
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.ensure_future(handler(data))
                else:
                    handler(data)
            except Exception as e:
                logger.error("Event handler error for {}: {}", event_type, e)

    async def connect(self) -> bool:
        """Connect to AIRI WebSocket server.

        Returns:
            True if connected successfully.
        """
        try:
            logger.info("Connecting to AIRI at {}", self.url)
            self._ws = await websockets.connect(
                self.url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5,
            )
            self._connected = True
            logger.info("Connected to AIRI")

            # Send authentication if token is set
            if self.token:
                await self._send_authenticate()

            return True

        except (websockets.WebSocketException, OSError) as e:
            logger.error("Failed to connect to AIRI: {}", e)
            self._connected = False
            return False

    async def _send_authenticate(self) -> None:
        """Send authentication message."""
        auth_msg = {
            "type": "module:authenticate",
            "data": {
                "token": self.token,
            },
        }
        await self.send(auth_msg)
        logger.debug("Authentication sent")

    async def disconnect(self) -> None:
        """Disconnect from AIRI."""
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("Disconnected from AIRI")

    async def send(self, message: dict) -> bool:
        """Send a JSON message to AIRI.

        Args:
            message: Dictionary to send as JSON.

        Returns:
            True if sent successfully.
        """
        if not self._ws or not self._connected:
            logger.warning("Not connected, cannot send message")
            return False

        try:
            payload = json.dumps(message, ensure_ascii=False)
            await self._ws.send(payload)
            return True
        except websockets.WebSocketException as e:
            logger.error("Send error: {}", e)
            self._connected = False
            return False

    async def _recv_loop(self) -> None:
        """Receive and dispatch messages from AIRI."""
        logger.info("Receive loop started")

        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                    event_type = message.get("type", "")
                    data = message.get("data", {})

                    if event_type:
                        self._dispatch_event(event_type, data)

                        # Log message types (debug)
                        if event_type != AIRIEventType.TRANSPORT_HEARTBEAT:
                            logger.debug("Received event: {} | data_keys={}",
                                         event_type, list(data.keys()))

                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON from AIRI: {}", e)

        except websockets.WebSocketException as e:
            logger.error("Receive loop error: {}", e)
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        finally:
            self._connected = False
            logger.info("Receive loop ended")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to AIRI."""
        try:
            while self._running and self._connected:
                await asyncio.sleep(30)
                if self._connected:
                    await self.send({
                        "type": "transport:connection:heartbeat",
                        "data": {
                            "kind": "ping",
                            "message": "🩵",
                        },
                    })
        except asyncio.CancelledError:
            pass

    async def run(self) -> None:
        """Run the AIRI client with auto-reconnect."""
        self._running = True
        attempt = 0

        while self._running:
            # Connect
            success = await self.connect()

            if not success:
                attempt += 1
                if self.max_attempts > 0 and attempt >= self.max_attempts:
                    logger.error("Max reconnect attempts ({}) reached",
                                 self.max_attempts)
                    break

                logger.info("Reconnecting in {}s (attempt {}/{})",
                            self.reconnect_interval, attempt,
                            self.max_attempts or "∞")
                await asyncio.sleep(self.reconnect_interval)
                continue

            # Connected - reset attempt counter
            attempt = 0

            # Start receive and heartbeat
            recv_task = asyncio.create_task(self._recv_loop())
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Wait for disconnection
            try:
                await recv_task
            except asyncio.CancelledError:
                pass
            finally:
                heartbeat_task.cancel()
                await self.disconnect()

            # Reconnect loop
            if self._running:
                logger.info("Reconnecting in {}s...", self.reconnect_interval)
                await asyncio.sleep(self.reconnect_interval)

    async def stop(self) -> None:
        """Stop the AIRI client."""
        self._running = False
        await self.disconnect()
        logger.info("AIRI client stopped")

    async def send_input_text(self, text: str, **overrides) -> bool:
        """Send an input:text event to AIRI.

        Args:
            text: Text content.
            **overrides: Optional overrides (sessionId, etc.).

        Returns:
            True if sent successfully.
        """
        message = {
            "type": "input:text",
            "data": {
                "text": text,
                **overrides,
            },
        }
        return await self.send(message)

    async def send_input_text_voice(
        self,
        text: str,
        audio: bytes | None = None,
        **overrides,
    ) -> bool:
        """Send an input:text:voice event to AIRI.

        Args:
            text: Transcribed text.
            audio: Optional audio data bytes.
            **overrides: Optional overrides.

        Returns:
            True if sent successfully.
        """
        message = {
            "type": "input:text:voice",
            "data": {
                "text": text,
                **overrides,
            },
        }
        if audio is not None:
            message["data"]["audio"] = audio.hex() if isinstance(audio, bytes) else audio

        return await self.send(message)
