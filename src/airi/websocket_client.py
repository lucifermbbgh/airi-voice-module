"""
AIRI WebSocket client module.

Connects to AIRI's plugin protocol WebSocket server and communicates
using the Eventa-based event system.

Protocol reference (verified against moeru-ai/airi
`packages/plugin-protocol/src/types/events.ts` and
`packages/server-sdk/src/client.ts`):

- Handshake: (optional) module:authenticate → module:announce → module:announced
- Every outgoing message carries `metadata.source` (ModuleIdentity) and
  `metadata.event.id`.
- `input:text:voice` data field is `transcription` (NOT `text`).
- `output:gen-ai:chat:message` data field is `message` (AssistantMessage whose
  text lives in `message.content`), not a flat `text`.

Phase 1: Basic connection, authentication, heartbeat, and event listening.
Future phases: Send input:text:voice events, receive TTS responses.
"""

from __future__ import annotations

import asyncio
import json
import uuid
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
    OUTPUT_CHAT_TOOL_CALL = "output:gen-ai:chat:tool-call"
    TRANSPORT_HEARTBEAT = "transport:connection:heartbeat"
    ERROR = "error"


# Control message types handled internally (not dispatched to handlers).
_MODULE_AUTHENTICATED = "module:authenticated"
_MODULE_ANNOUNCED = "extension:module:announced"
_REGISTRY_MODULES_SYNC = "registry:modules:sync"


class AIRIClient:
    """AIRI plugin protocol WebSocket client.

    Attributes:
        host: AIRI server hostname.
        port: AIRI WebSocket port.
        path: WebSocket path.
        token: Authentication token.
        reconnect_interval: Seconds between reconnect attempts.
        max_attempts: Maximum reconnect attempts (0 = unlimited).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6121,
        path: str = "/ws",
        token: str = "",
        name: str = "voice-module",
        reconnect_interval: int = 5,
        max_attempts: int = 0,
    ):
        """Initialize AIRI WebSocket client.

        Args:
            host: AIRI server hostname.
            port: WebSocket port.
            path: WebSocket path.
            token: Authentication token.
            name: Plugin/module name (stable identifier across instances).
            reconnect_interval: Seconds between reconnects.
            max_attempts: Max reconnect attempts (0 = unlimited).
        """
        self.host = host
        self.port = port
        self.path = path
        self.token = token
        self.name = name
        self.reconnect_interval = reconnect_interval
        self.max_attempts = max_attempts

        # Unique instance id (per process/deployment), matches ModuleIdentity.id.
        self._instance_id = f"{self.name}-{uuid.uuid4().hex[:8]}"

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._running = False
        self._connected = False
        self._ready = False  # True once module:announced received
        self._event_handlers: dict[str, list[callable]] = {}

    @property
    def url(self) -> str:
        """Get WebSocket URL."""
        return f"ws://{self.host}:{self.port}{self.path}"

    @property
    def is_connected(self) -> bool:
        """Check if connected to AIRI."""
        return self._connected

    @property
    def is_ready(self) -> bool:
        """Check if the module handshake (announce) has completed."""
        return self._ready

    @property
    def _identity(self) -> dict:
        """ExtensionModuleIdentity (AIRI 0.11.3): {id, extension: {id}}."""
        return {
            "id": self._instance_id,
            "extension": {"id": self.name},
        }

    @property
    def _source(self) -> dict:
        """Metadata source (0.11.3): kind + identity + legacy plugin field."""
        return {
            "kind": "plugin",
            "id": self._instance_id,
            "extension": {"id": self.name},
            "plugin": {"id": self.name},
        }

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
            self._ready = False
            logger.info("Connected to AIRI")
            return True

        except (websockets.WebSocketException, OSError) as e:
            logger.error("Failed to connect to AIRI: {}", e)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from AIRI."""
        self._connected = False
        self._ready = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("Disconnected from AIRI")

    async def send(self, message: dict) -> bool:
        """Send a JSON message to AIRI.

        Automatically attaches `metadata.source` (ModuleIdentity) and
        `metadata.event.id`, as required by the AIRI protocol.

        Args:
            message: Dictionary with at least "type" and "data" keys.

        Returns:
            True if sent successfully.
        """
        if not self._ws or not self._connected:
            logger.warning("Not connected, cannot send message")
            return False

        payload = {
            "type": message["type"],
            "data": message.get("data", {}),
            "metadata": {
                "source": self._source,
                "event": {"id": uuid.uuid4().hex[:16]},
            },
        }

        try:
            raw = json.dumps(payload, ensure_ascii=False)
            await self._ws.send(raw)
            return True
        except websockets.WebSocketException as e:
            logger.error("Send error: {}", e)
            self._connected = False
            return False

    # ── Handshake ────────────────────────────────────────────────

    async def _authenticate(self) -> None:
        """Send module:authenticate (only when a token is configured)."""
        await self.send({
            "type": "module:authenticate",
            "data": {"token": self.token},
        })
        logger.debug("Authentication sent")

    async def _announce(self) -> None:
        """Send extension:module:announce to declare this module to AIRI."""
        await self.send({
            "type": "extension:module:announce",
            "data": {
                "name": self.name,
                "identity": self._identity,
                "possibleEvents": [],
                "dependencies": [],
            },
        })
        logger.debug("Module announce sent: {} ({})", self.name, self._instance_id)

    async def _start_handshake(self) -> None:
        """Begin the handshake after the socket is open."""
        if self.token:
            await self._authenticate()
        else:
            await self._announce()

    async def _handle_control_message(self, event_type: str, data: dict) -> bool:
        """Handle protocol control messages.

        Returns True if the message was consumed (should not be dispatched
        to user handlers), False otherwise.
        """
        if event_type == _MODULE_AUTHENTICATED:
            if data.get("authenticated"):
                logger.info("AIRI authentication succeeded")
                await self._announce()
            else:
                logger.error("AIRI authentication failed")
            return True

        if event_type == _MODULE_ANNOUNCED:
            identity = data.get("identity") or {}
            if data.get("name") == self.name and identity.get("id") == self._instance_id:
                if not self._ready:
                    self._ready = True
                    logger.info("AIRI module announced, ready ({})", self._instance_id)
            return True

        if event_type == _REGISTRY_MODULES_SYNC:
            # Fallback: if announce succeeded but module:announced was missed.
            if self._ready:
                return True
            modules = data.get("modules") or []
            for m in modules:
                mid = (m.get("identity") or {}) if isinstance(m, dict) else {}
                if m.get("name") == self.name and mid.get("id") == self._instance_id:
                    self._ready = True
                    logger.info("AIRI module ready via registry sync")
                    break
            return True

        if event_type == AIRIEventType.ERROR:
            logger.error("AIRI error event: {}", data.get("message") or data)
            return True

        if event_type == AIRIEventType.TRANSPORT_HEARTBEAT:
            # Respond to server pings with a pong.
            if data.get("kind") == "ping":
                await self.send({
                    "type": "transport:connection:heartbeat",
                    "data": {"kind": "pong", "message": "💛"},
                })
            return True

        return False

    @staticmethod
    def _parse_message(raw: str) -> dict:
        """Parse an incoming message (JSON, with a superjson fallback)."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON message: {raw[:100]}") from e

        # superjson envelope: {"json": <payload>, "meta": {...}?}
        # AIRI emits {"json": {...}} even without special values; meta is optional.
        if isinstance(parsed, dict) and isinstance(parsed.get("json"), dict):
            return parsed["json"]

        return parsed

    async def _recv_loop(self) -> None:
        """Receive and dispatch messages from AIRI."""
        logger.info("Receive loop started")

        try:
            async for raw in self._ws:
                try:
                    message = self._parse_message(raw)
                except ValueError as e:
                    logger.warning("Dropping malformed message: {}", e)
                    continue

                event_type = message.get("type", "")
                data = message.get("data", {}) or {}

                if not event_type:
                    continue

                if await self._handle_control_message(event_type, data):
                    continue

                self._dispatch_event(event_type, data)

                if event_type != AIRIEventType.TRANSPORT_HEARTBEAT:
                    logger.debug(
                        "Received event: {} | data_keys={}",
                        event_type, list(data.keys()) if isinstance(data, dict) else [],
                    )

        except websockets.WebSocketException as e:
            logger.error("Receive loop error: {}", e)
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        finally:
            self._connected = False
            self._ready = False
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

            # Start handshake after the receive loop is running so the
            # module:announced response can be received.
            await self._start_handshake()

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

    async def send_input_text_voice(self, text: str, **overrides) -> bool:
        """Send an input:text:voice event to AIRI.

        Args:
            text: Transcribed text.
            **overrides: Optional overrides.

        Returns:
            True if sent successfully.
        """
        message = {
            "type": "input:text:voice",
            "data": {
                "transcription": text,
                **overrides,
            },
        }
        return await self.send(message)
